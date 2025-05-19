# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.distributed as dist
import math
import numpy as np
from collections import defaultdict


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ClassAwareBatchSampler(torch.utils.data.Sampler):
    """
    ClassAwareBatchSampler with Batch-Softmax Balance
    
    每个批次包含N个类，每个类K个样本，共N*K个样本
    可以选择基于历史loss或softmax熵动态调整类别采样权重
    
    Args:
        dataset: 数据集对象（需要有targets或get_labels方法）
        batch_size: 批次大小
        num_classes_per_batch: 每个批次选择的类别数量(N)
        num_samples_per_class: 每个类别选择的样本数量(K)
        use_softmax_balance: 是否使用softmax熵平衡（默认False）
        class_weights: 可选的自定义类别权重
        temperature: softmax权重计算的温度系数
        fallback_to_random: 当无法构建完整批次时是否回退到随机采样
        seed: 随机种子
    """
    def __init__(
        self, 
        dataset,
        num_classes_per_batch=8,
        num_samples_per_class=8,
        use_softmax_balance=False,
        class_weights=None,
        temperature=1.0,
        fallback_to_random=True,
        seed=0
    ):
        self.dataset = dataset
        self.num_classes_per_batch = num_classes_per_batch
        self.num_samples_per_class = num_samples_per_class
        self.use_softmax_balance = use_softmax_balance
        self.temperature = temperature
        self.fallback_to_random = fallback_to_random
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.epoch = 0
        
        # 获取数据集标签
        if hasattr(dataset, 'targets'):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, 'get_labels'):
            self.labels = np.array(dataset.get_labels())
        else:
            raise AttributeError("数据集必须有'targets'属性或'get_labels'方法")
            
        # 类别计数和索引映射
        self.num_classes = len(np.unique(self.labels))
        print(f"ClassAwareBatchSampler: 数据集包含 {self.num_classes} 个类别")
        
        # 验证参数
        if self.num_classes < self.num_classes_per_batch:
            print(f"Warning: 数据集类别数({self.num_classes})小于每批次类别数({self.num_classes_per_batch})，将使用所有可用类别")
            self.num_classes_per_batch = self.num_classes
            
        # 创建每个类别的样本索引
        self.class_indices = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.class_indices[label].append(i)
            
        # 验证每个类别的样本数量
        self.valid_classes = []
        for cls_id, indices in self.class_indices.items():
            if len(indices) >= self.num_samples_per_class:
                self.valid_classes.append(cls_id)
            else:
                print(f"Warning: 类别 {cls_id} 只有 {len(indices)} 个样本，少于所需的 {self.num_samples_per_class}，将被排除")
                
        if len(self.valid_classes) < self.num_classes_per_batch:
            print(f"Warning: 只有 {len(self.valid_classes)} 个类别有足够样本，少于所需的 {self.num_classes_per_batch}")
            if len(self.valid_classes) == 0:
                print("Error: 没有类别有足够样本。将回退到随机采样")
                self.fallback_to_random = True
            else:
                self.num_classes_per_batch = len(self.valid_classes)
                
        # 设置类别权重
        if class_weights is not None:
            self.class_weights = np.array(class_weights)
        else:
            self.class_weights = np.ones(self.num_classes)
            
        # Softmax平衡的历史损失记录
        if use_softmax_balance:
            self.class_loss_history = np.ones(self.num_classes)  # 初始化为相等权重
            
        # 计算每个batch的大小和batch数量
        self.batch_size = self.num_classes_per_batch * self.num_samples_per_class
        # 设置epoch内的batch数量
        self.total_samples = len(dataset)
        self.num_batches = self.total_samples // self.batch_size
        
        print(f"ClassAwareBatchSampler: 每批次 {self.num_classes_per_batch} 类 x {self.num_samples_per_class} 样本")
        print(f"ClassAwareBatchSampler: 有效批次大小 = {self.batch_size}，批次数量 = {self.num_batches}")
    
    def __iter__(self):
        # 如果无法组成有效的采样，回退到随机采样
        if self.fallback_to_random and (len(self.valid_classes) < self.num_classes_per_batch or self.num_classes_per_batch <= 0):
            print("Warning: 无法构建类别平衡批次，回退到随机采样")
            indices = self.rng.permutation(len(self.dataset))
            for i in range(0, len(indices), self.batch_size):
                if i + self.batch_size <= len(indices):
                    yield indices[i:i + self.batch_size].tolist()
            return
        
        # 基于权重计算类别采样概率
        if self.use_softmax_balance:
            # 应用SoftMax，温度越高，分布越平坦
            exp_weights = np.exp(self.class_loss_history / self.temperature)
            class_probs = exp_weights / exp_weights.sum()
        else:
            # 使用预设权重
            class_probs = self.class_weights[self.valid_classes] / np.sum(self.class_weights[self.valid_classes])
        
        # 生成num_batches个batch
        for _ in range(self.num_batches):
            batch_indices = []
            
            # 按概率分布选择N个类别
            selected_classes = self.rng.choice(
                self.valid_classes, 
                size=self.num_classes_per_batch, 
                replace=False, 
                p=class_probs if len(class_probs) == len(self.valid_classes) else None
            )
            
            # 为每个选中类别抽取K个样本
            for cls_id in selected_classes:
                cls_indices = self.class_indices[cls_id]
                # 随机选择K个样本（不放回）
                if len(cls_indices) >= self.num_samples_per_class:
                    sampled_indices = self.rng.choice(cls_indices, size=self.num_samples_per_class, replace=False)
                else:
                    # 如果样本不足，允许重复采样
                    sampled_indices = self.rng.choice(cls_indices, size=self.num_samples_per_class, replace=True)
                
                batch_indices.extend(sampled_indices)
            
            # 打乱batch内样本顺序
            self.rng.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + epoch)
        
    def update_class_weights(self, class_losses):
        """
        更新类别权重，用于softmax平衡
        Args:
            class_losses: 每个类别的平均损失值
        """
        if self.use_softmax_balance:
            # 使用指数移动平均更新历史损失
            alpha = 0.9  # 平滑系数
            self.class_loss_history = alpha * self.class_loss_history + (1 - alpha) * class_losses
