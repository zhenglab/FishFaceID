# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements various loss functions, including CLIP-style contrastive losses.
"""
import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import math # Needed for ArcFace/CosFace

# ==========================================================
# 新增 FocalLoss 类
# ==========================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Args:
        alpha (float, optional): Weighting factor for balancing positive/negative examples.
            Defaults to 0.25.
        gamma (float, optional): Focusing parameter. Defaults to 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Defaults to 'mean'.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Defaults to -100.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): Logits from the model (before Softmax).
                                   Shape: [N, C] where N is batch size and C is number of classes.
            targets (torch.Tensor): Ground truth labels. Shape: [N].
        """
        # Filter out ignored targets
        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        if targets.numel() == 0: # Handle cases where all targets are ignored
            if self.reduction == 'mean':
                 return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            elif self.reduction == 'sum':
                 return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            else: # 'none'
                 # If input was empty, output should be empty too, but needs grad.
                 # Returning a tensor based on input shape before filtering is tricky.
                 # For 'none', perhaps return an empty tensor, but this might break things.
                 # A zero tensor is safer for aggregation later.
                 return torch.tensor(0.0, device=inputs.device, requires_grad=True)


        # Calculate Cross Entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)

        # Get probabilities of the true class
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss modulating factor
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha balancing factor
        # Get alpha weights for each sample based on its target class
        # Note: Assumes alpha is for the positive class (foreground)
        # In multi-class, this usually becomes a weight per class.
        # Simplified: apply alpha to the focal weight directly.
        # A common approach is alpha_t = alpha for foreground, 1-alpha for background.
        # Here, we use alpha directly on the focal weight, which is simpler
        # but might need adjustment based on how alpha is intended to be used.
        # If alpha is per-class:
        # alpha_t = torch.ones_like(targets, dtype=torch.float, device=inputs.device) * (1 - self.alpha)
        # alpha_t[targets == positive_class_index] = self.alpha # Needs positive_class_index
        # For multi-class, often a fixed alpha is applied to all classes equally,
        # or alpha is a vector of weights per class. Let's use a fixed alpha for simplicity.
        alpha_weight = self.alpha # Using a single alpha value

        # Combine weights
        loss = alpha_weight * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    NOTE: The base_criterion should be compatible with the student model's output format.
    If using similarity-based losses below, ensure compatibility or wrap appropriately.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs

        # Calculate base loss - Ensure base_criterion is appropriate for 'outputs'
        # If base_criterion expects features (like InfoNCE below), 'outputs' must be features.
        # If base_criterion expects logits (like CE), 'outputs' must be logits.
        try:
            if isinstance(self.base_criterion, (InfoNCELoss, CosineSimilarityLoss)):
                 # These might expect features, handle potential mismatch if 'outputs' aren't features
                 # This part might need adjustment based on actual model output and base_criterion needs
                 # Assuming 'outputs' holds the primary data needed by base_criterion for now.
                 # If base needs image+prompt features, this forward signature is insufficient.
                 print("Warning: Using feature-based loss within DistillationLoss. Ensure 'outputs' format is correct.")
                 # Placeholder: This likely needs refinement based on how features are passed.
                 # If outputs = (logits, features), maybe: base_loss = self.base_criterion(features[0], features[1], labels)?
                 base_loss = self.base_criterion(outputs, labels) # This might fail if base_criterion expects features
            elif isinstance(self.base_criterion, TripletLoss):
                 # TripletLoss expects anchor, positive, negative. 'outputs' format must match.
                 print("Warning: Using TripletLoss within DistillationLoss. Ensure 'outputs' format provides anchor/positive/negative.")
                 # Placeholder: base_loss = self.base_criterion(outputs[0], outputs[1], outputs[2])?
                 base_loss = self.base_criterion(outputs, labels) # This might fail.
            else: # Assume standard losses like CE
                 base_loss = self.base_criterion(outputs, labels)
        except Exception as e:
             print(f"Error calculating base loss in DistillationLoss: {e}")
             print("Check compatibility between DistillationLoss input 'outputs' and base_criterion expectations.")
             raise e


        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
             # Standard distillation typically uses the distillation token's output (logits).
             # Modify this if your distillation target comes from features.
            raise ValueError("When knowledge distillation is enabled, the model is "
                              "expected to return a Tuple[Tensor, Tensor] with the main output "
                              "and the distillation predictions (usually logits).")

        # don't backprop through the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs) # Teacher output usually logits

        # KD Loss calculation usually compares student's distillation output (logits) with teacher's output (logits)
        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    Requires anchor, positive, and negative samples as input features.
    支持基本三元组损失和Batch Hard Mining策略。
    """
    def __init__(self, margin=1.0, reduction='mean', mining='random'):
        """
        Args:
            margin: 三元组损失的间隔参数
            reduction: 损失计算的降维方式，'mean'|'sum'|'none'
            mining: 三元组挖掘策略，'random'|'hard'|'semi-hard'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.mining = mining
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='none') # Standard Euclidean distance

    def forward(self, anchor, positive, negative):
        """
        标准三元组损失的前向传播，用于预先构建的三元组
        Args:
            anchor: 锚点样本特征 (shape: [N, D])
            positive: 正样本特征 (shape: [N, D])
            negative: 负样本特征 (shape: [N, D])
        """
        loss = self.triplet_loss(anchor, positive, negative)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
             return loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

    def batch_hard_mining(self, features, labels):
        """
        执行Batch Hard Mining，为每个样本找到最硬正样本和最硬负样本
        Args:
            features: 批次中的样本特征 (shape: [B, D])
            labels: 样本标签 (shape: [B])
        Returns:
            计算的三元组损失
        """
        # 确保输入是二维张量，且维度匹配
        if features.dim() != 2:
            raise ValueError(f"Features should be 2D tensor, got {features.dim()}D")
        
        batch_size = features.size(0)
        device = features.device
        
        # 规范化特征，计算余弦相似度矩阵
        features_norm = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(features_norm, features_norm.t())  # [B, B]
        
        # 转换为距离矩阵 (1 - 相似度)
        dist_matrix = 1.0 - sim_matrix  # [B, B]
        
        # 创建标签相等的掩码，用于区分正样本和负样本
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        labels_not_equal = ~labels_equal  # [B, B]
        
        # 对角线掩码 (排除自身)
        diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)  # [B, B]
        
        # 正样本掩码 (同类但非自身)
        positive_mask = labels_equal & diag_mask  # [B, B]
        
        # 负样本掩码 (不同类)
        negative_mask = labels_not_equal  # [B, B]
        
        # 根据mining策略选择正负样本
        if self.mining == 'hard':
            # 硬挖掘: 每个锚点选择最远的正样本和最近的负样本
            
            # 确保每个样本至少有一个正样本和一个负样本
            has_pos = positive_mask.any(dim=1)
            has_neg = negative_mask.any(dim=1)
            valid_samples = has_pos & has_neg
            
            if not valid_samples.any():
                # 如果没有有效样本，返回零损失
                return torch.tensor(0.0, device=device)
            
            # 对于无效样本位置设置极大/极小值，以便在argmax/argmin时被忽略
            pos_dist = dist_matrix.clone()
            neg_dist = dist_matrix.clone()
            
            # 设置掩码外的值为很小的值(负距离)，确保不会被argmax选中
            pos_dist[~positive_mask] = -1.0
            
            # 设置掩码外的值为很大的值，确保不会被argmin选中
            neg_dist[~negative_mask] = 2.0  # 余弦距离最大为2
            
            # 为每个锚点找到最难的正样本和负样本
            hardest_pos_idx = pos_dist.argmax(dim=1)  # [B]
            hardest_neg_idx = neg_dist.argmin(dim=1)  # [B]
            
            # 只保留有效样本
            valid_anchors = features[valid_samples]
            valid_positives = features[hardest_pos_idx[valid_samples]]
            valid_negatives = features[hardest_neg_idx[valid_samples]]
            
            # 计算三元组损失
            loss = self.triplet_loss(valid_anchors, valid_positives, valid_negatives)
            
        elif self.mining == 'semi-hard':
            # 半硬挖掘: 选择比锚点和正样本距离大一点但不是最大的负样本
            
            # 为每个锚点计算损失矩阵
            loss_matrix = torch.zeros((batch_size, batch_size, batch_size), device=device)
            
            # 筛选有效的样本
            has_pos = positive_mask.any(dim=1)
            has_neg = negative_mask.any(dim=1)
            valid_samples = has_pos & has_neg
            
            if not valid_samples.any():
                # 如果没有有效样本，返回零损失
                return torch.tensor(0.0, device=device)
            
            # 选择锚点
            for i in range(batch_size):
                if not valid_samples[i]:
                    continue
                    
                # 获取正样本索引
                pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
                
                # 对每个正样本
                for p_idx in pos_indices:
                    # 获取负样本索引
                    neg_indices = negative_mask[i].nonzero(as_tuple=True)[0]
                    
                    # 计算锚点到正样本的距离
                    ap_dist = dist_matrix[i, p_idx]
                    
                    # 计算锚点到所有负样本的距离
                    an_dists = dist_matrix[i, neg_indices]
                    
                    # 选择满足半硬条件的负样本：距离大于ap_dist但不是最大的
                    semi_hard_negs = (an_dists > ap_dist) & (an_dists < ap_dist + self.margin)
                    
                    if semi_hard_negs.any():
                        # 有半硬负样本，随机选择一个
                        semi_hard_idx = torch.where(semi_hard_negs)[0]
                        selected_neg_idx = neg_indices[semi_hard_idx[torch.randint(len(semi_hard_idx), (1,), device=device)]]
                    else:
                        # 没有半硬负样本，选择最近的负样本(硬负样本)
                        selected_neg_idx = neg_indices[an_dists.argmin()]
                    
                    # 计算损失并存储
                    triplet_loss = self.triplet_loss(features[i:i+1], features[p_idx:p_idx+1], features[selected_neg_idx:selected_neg_idx+1])
                    loss_matrix[i, p_idx, selected_neg_idx] = triplet_loss
            
            # 获取所有有效的损失
            valid_loss = loss_matrix[loss_matrix > 0]
            
            if len(valid_loss) == 0:
                return torch.tensor(0.0, device=device)
            
            loss = valid_loss
        else: # 'random' 或未指定
            # 随机挖掘(默认行为)：随机选择正负样本
            # 这里我们不会真正执行random mining，因为该功能已在CombinedLoss中实现
            # 返回一个占位损失，实际不使用
            loss = torch.tensor(0.0, device=device)
        
        # 应用reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss


class InfoNCELoss(nn.Module):
    """
    Calculates the InfoNCE loss for contrastive learning (asymmetric case).
    Typically used for image batch vs. class prompts.
    Assumes image_features and prompt_features are L2 normalized.
    
    支持两种输入格式:
    1. prompt_features形状为(C, D) - 每个类别一个prompt
    2. prompt_features形状为(C, P, D) - 每个类别P个prompt
    """
    def __init__(self, temperature=0.07, reduction='mean'):
        """
        Args:
            temperature (float): Temperature scaling factor.
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, image_features, prompt_features, labels):
        """
        Args:
            image_features (torch.Tensor): Batch of image features (N, D). Assumed normalized.
            prompt_features (torch.Tensor): 
                - 单prompt格式: 类别prompt特征矩阵(C, D)
                - 多prompt格式: 类别prompt特征张量(C, P, D), P是每类的prompt数量
            labels (torch.Tensor): Ground truth labels (N), integer indices corresponding to rows in prompt_features.

        Returns:
            torch.Tensor: The InfoNCE loss.
        """
        # 确保输入特征已经归一化，创建副本避免修改原始输入
        image_features = F.normalize(image_features.clone(), dim=-1)
        
        # 处理prompt_features不同的形状
        if prompt_features.dim() == 2:
            # 传统格式: (C, D) - 每类一个prompt
            # 创建归一化副本
            prompt_features_norm = F.normalize(prompt_features.clone(), dim=-1)
            
            if image_features.shape[1] != prompt_features_norm.shape[1]:
                raise ValueError(f"Embedding dimension mismatch: image_features D={image_features.shape[1]}, prompt_features D={prompt_features_norm.shape[1]}")
                
            # 计算logits: (N, D) @ (D, C) -> (N, C)
            logits = image_features @ prompt_features_norm.t()
            
        elif prompt_features.dim() == 3:
            # 多prompt格式: (C, P, D) - 每类P个prompt
            C, P, D = prompt_features.shape
            
            if image_features.shape[1] != D:
                raise ValueError(f"Embedding dimension mismatch: image_features D={image_features.shape[1]}, prompt_features D={D}")
            
            # 创建归一化副本
            prompt_features_norm = F.normalize(prompt_features.clone(), dim=-1)
            
            # 两种策略处理多prompt:
            # 1. 选择与样本最相似的prompt (max similarity)
            # 创建新张量重塑prompt_features为(C*P, D)以计算所有相似度
            reshaped_prompts = prompt_features_norm.reshape(-1, D)  # (C*P, D)
            
            # 计算每个样本与所有prompt的相似度
            all_similarities = image_features @ reshaped_prompts.t()  # (N, C*P)
            
            # 将相似度矩阵重塑为(N, C, P)以找到每类最相似的prompt
            all_similarities = all_similarities.reshape(-1, C, P)  # (N, C, P)
            
            # 对每个类别，选择最相似的prompt
            logits, _ = all_similarities.max(dim=2)  # (N, C)
            
        else:
            raise ValueError(f"prompt_features 必须是2D (C, D)或3D (C, P, D)张量, 实际形状: {prompt_features.shape}")

        # 应用温度缩放
        logits = logits / self.temperature

        # 使用提供的标签计算交叉熵损失
        # 标签应该是0到C-1的整数
        loss = F.cross_entropy(logits, labels, reduction='none')

        # 应用损失归约方式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class CosineSimilarityLoss(nn.Module):
    """
    Calculates loss based on maximizing cosine similarity between positive pairs.
    Loss = 1 - cosine_similarity.
    Note: This loss alone might not be sufficient for strong representation learning
    as it doesn't explicitly handle negative pairs. InfoNCE is generally preferred.
    """
    def __init__(self, reduction='mean', dim=1):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            dim (int): Dimension along which to compute cosine similarity.
        """
        super().__init__()
        self.reduction = reduction
        self.dim = dim

    def forward(self, features1, features2):
        """
        Args:
            features1 (torch.Tensor): Batch of features (e.g., image features) (N, D).
            features2 (torch.Tensor): Batch of corresponding features (e.g., prompt features) (N, D).

        Returns:
            torch.Tensor: The cosine similarity loss.
        """
        # Using F.cosine_similarity handles normalization internally
        sim = F.cosine_similarity(features1, features2, dim=self.dim)
        # Loss is 1 - similarity (maximizes similarity towards 1)
        loss = 1.0 - sim

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss."""
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False, reduction='mean', use_focal_loss=False, focal_gamma=2.0, focal_alpha=0.25): # 新增 Focal Loss 相关参数
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.reduction = reduction
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # ==========================================================
        # 选择使用 Focal Loss 还是 Cross Entropy
        # ==========================================================
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            print(f"ArcFaceLoss using Focal Loss with gamma={focal_gamma}, alpha={focal_alpha}")
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
        else:
            print("ArcFaceLoss using Cross Entropy Loss")
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        # ==========================================================

    def to(self, device):
        """Moves the loss module and its parameters to the specified device."""
        super().to(device)
        self.weight = self.weight.to(device) # Ensure weight is on the correct device
        # If criterion has parameters (like FocalLoss doesn't usually, but some might), move them too
        self.criterion.to(device)
        return self

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim).
            targets: ground truth labels with shape (batch_size).
        """
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1) # clamp for numerical stability
        phi = cosine * self.cos_m - sine * self.sin_m # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Original ArcFace implementation condition
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert targets to one-hot format
        one_hot = torch.zeros(cosine.size(), device=inputs.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        # Apply margin to the correct class logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # ==========================================================
        # 使用选择的损失函数计算损失
        # ==========================================================
        loss = self.criterion(output, targets)
        # ==========================================================

        return loss


class CosFaceLoss(nn.Module):
    """ Additive Cosine Margin Loss """
    def __init__(self, in_features, out_features, s=64.0, m=0.35, reduction='mean'):
        super(CosFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.reduction = reduction
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def to(self, device):
        super(CosFaceLoss, self).to(device)
        return self

    def forward(self, inputs, targets):
        """ inputs: features (N, D), targets: labels (N) """
        # 确保targets在正确的设备上
        if targets.device != inputs.device:
            targets = targets.to(inputs.device)
            
        # 确保权重在正确的设备上
        if self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)
            
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        
        # Margin cosine: cosine - m
        phi = cosine - self.m
        
        # 创建one-hot张量
        one_hot = torch.zeros(cosine.size(), device=inputs.device)
        # 使用不修改原张量的方式创建one_hot编码
        targets_reshaped = targets.reshape(-1, 1).long()
        one_hot = one_hot.scatter(1, targets_reshaped, 1.0)
        
        output = torch.where(one_hot.bool(), phi, cosine)
        output *= self.s
        
        loss = F.cross_entropy(output, targets, reduction=self.reduction)
        return loss


class CenterLoss(nn.Module):
    """ Computes the Center Loss """
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        # Learnable class centers
        self.centers = nn.Parameter(torch.zeros(self.num_classes, self.feat_dim).to(device))

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): Features of shape (batch_size, feat_dim).
            labels (Tensor): Ground truth labels, LongTensor of shape (batch_size).
        """
        batch_size = features.size(0)
        
        # 确保labels在正确的设备上
        if labels.device != features.device:
            labels = labels.to(features.device)
            
        # 添加特征归一化
        features = F.normalize(features, p=2, dim=1)
        centers_normalized = F.normalize(self.centers, p=2, dim=1)
        
        # 选择对应的中心
        center_selected = centers_normalized[labels]
        
        # 使用余弦距离代替欧氏距离，提高稳定性
        # 余弦相似度在[-1,1]之间，转换为距离在[0,2]之间
        sim = torch.sum(features * center_selected, dim=1)
        dist = 1.0 - sim
        
        # 应用损失截断，防止极端值
        dist = torch.clamp(dist, min=0.0, max=2.0)
        
        # 计算损失
        loss = torch.mean(dist)
        return loss


class IntermediateLayerLoss(nn.Module):
    """
    为模型的中间层特征提供监督信号的辅助损失。
    支持多种损失类型：对比损失(contrastive)、KL散度(kl)和MSE(mse)
    """
    def __init__(self, embed_dim, num_classes, loss_type='contrastive', 
                 temperature=0.07, reduction='mean', device='cuda'):
        """
        Args:
            embed_dim: 特征维度
            num_classes: 类别数量
            loss_type: 损失类型，支持'contrastive'、'kl'和'mse'
            temperature: 对比损失的温度参数
            reduction: 损失计算的降维方式，'mean'|'sum'|'none'
            device: 设备
        """
        super(IntermediateLayerLoss, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.temperature = temperature
        self.reduction = reduction
        self.device = device
        
        # 为对比损失初始化可学习的类别代理特征
        if loss_type == 'contrastive':
            self.class_proxies = nn.Parameter(torch.randn(num_classes, embed_dim).to(device) * 0.02)
            nn.init.xavier_normal_(self.class_proxies)
    
    def forward(self, features, labels, target_features=None):
        """
        计算中间层特征的损失
        
        Args:
            features: 中间层特征，形状为[N, D]
            labels: 类别标签，形状为[N]
            target_features: 目标特征(可选)，用于KL散度和MSE损失，形状为[N, D]
        
        Returns:
            计算得到的损失值
        """
        if self.loss_type == 'contrastive':
            return self._contrastive_loss(features, labels)
        elif self.loss_type == 'kl':
            if target_features is None:
                raise ValueError("KL散度损失需要提供target_features")
            return self._kl_loss(features, target_features)
        elif self.loss_type == 'mse':
            if target_features is None:
                raise ValueError("MSE损失需要提供target_features")
            return self._mse_loss(features, target_features)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
    
    def _contrastive_loss(self, features, labels):
        """对比损失 - 使用类别代理特征"""
        # 归一化特征和代理
        features = F.normalize(features, p=2, dim=1)
        proxies = F.normalize(self.class_proxies, p=2, dim=1)
        
        # 计算特征与所有代理的相似度
        logits = features @ proxies.t() / self.temperature
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels.long(), reduction=self.reduction)
        return loss
    
    def _kl_loss(self, features, target_features):
        """KL散度损失 - 使特征分布接近目标分布"""
        # 将特征转换为概率分布
        features_dist = F.log_softmax(features, dim=1)
        target_dist = F.softmax(target_features, dim=1)
        
        # 计算KL散度
        loss = F.kl_div(features_dist, target_dist, reduction=self.reduction)
        return loss
    
    def _mse_loss(self, features, target_features):
        """MSE损失 - 直接匹配特征"""
        # 归一化特征以保持一致性
        features = F.normalize(features, p=2, dim=1)
        target_features = F.normalize(target_features, p=2, dim=1)
        
        # 计算MSE损失
        loss = F.mse_loss(features, target_features, reduction=self.reduction)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, embed_dim, num_classes,
                 main_loss_type='infonce', # 'infonce', 'cosine_sim', 'ce', or 'ce_infonce'
                 aux_losses=None, # List, e.g., ['center', 'arcface']
                 aux_weights=None, # Dict, e.g., {'center': 0.5, 'arcface': 1.0}
                 infonce_temp=0.07,
                 triplet_margin=1.0, # Needed if triplet is used
                 triplet_mining='random', # 'random', 'hard', 'semi-hard'
                 center_alpha=0.5, # Learning rate for center updates (often handled separately)
                 arcface_s=64.0, arcface_m=0.50,
                 cosface_s=64.0, cosface_m=0.35,
                 contrast_weight=0.5, # 新增参数：InfoNCE在ce_infonce组合中的权重
                 # 中间层辅助损失参数
                 intermediate_loss_type='contrastive', # 'contrastive', 'kl', 'mse'
                 intermediate_temp=0.07,
                 device='cuda', **kwargs):
        super().__init__()
        self.main_loss_type = main_loss_type
        self.aux_losses_config = aux_losses if aux_losses is not None else []
        self.aux_weights = aux_weights if aux_weights is not None else {}
        self.device = device
        self.prompt_features = None # Store prompt features if needed by aux losses
        self.triplet_mining = triplet_mining
        self.contrast_weight = contrast_weight # 存储对比损失权重
        
        # 添加日志控制变量
        self.triplet_fallback_logged = False  # 是否已经记录过fallback警告
        self.eval_mode_logged = False  # 是否已经记录过评估模式警告

        print(f"Initializing CombinedLoss with main loss: {main_loss_type}")
        
        # 辅助函数 - 确保张量在相同设备上
        def ensure_same_device(tensor1, tensor2):
            """确保两个张量在同一设备上"""
            if tensor1.device != tensor2.device:
                return tensor2.to(tensor1.device)
            return tensor2
        
        self.ensure_same_device = ensure_same_device

        # --- Instantiate Main Loss ---
        if main_loss_type == 'infonce':
            self.main_loss_fn = InfoNCELoss(temperature=infonce_temp)
            print(f"  - InfoNCE Temp: {infonce_temp}")
        elif main_loss_type == 'cosine_sim':
             self.main_loss_fn = CosineSimilarityLoss()
             print("  - Using Cosine Similarity as main loss.")
        elif main_loss_type == 'ce':
             self.main_loss_fn = nn.CrossEntropyLoss()
             print("  - Using Cross Entropy as main loss.")
        elif main_loss_type == 'ce_infonce':
             # 初始化两个损失函数
             self.ce_loss_fn = nn.CrossEntropyLoss()
             self.infonce_loss_fn = InfoNCELoss(temperature=infonce_temp)
             print(f"  - Using CE+InfoNCE combined loss with contrast weight: {contrast_weight}")
             print(f"  - InfoNCE Temp: {infonce_temp}")
        else:
            raise ValueError(f"Unsupported main_loss_type: {main_loss_type}. Choose from 'infonce', 'cosine_sim', 'ce', 'ce_infonce'.")

        # --- Auxiliary Loss Restriction Check ---
        if main_loss_type not in ['infonce', 'ce_infonce'] and self.aux_losses_config:
             print(f"Warning: Auxiliary losses ({self.aux_losses_config}) are only supported when main_loss_type is 'infonce' or 'ce_infonce'. Ignoring auxiliary losses.")
             self.aux_losses_config = []
             self.aux_weights = {}

        # --- Instantiate Auxiliary Losses (Only if main_loss_type is 'infonce' or 'ce_infonce') ---
        self.aux_loss_fns = nn.ModuleDict()
        if self.aux_losses_config: # This check ensures it only runs if main_loss_type was valid
            print(f"Auxiliary losses enabled: {self.aux_losses_config}")
            print(f"Auxiliary weights: {self.aux_weights}")
            for loss_name in self.aux_losses_config:
                weight = self.aux_weights.get(loss_name, 1.0) # Default weight 1.0
                print(f"  - Adding Aux Loss: {loss_name} with weight {weight}")

                if loss_name == 'triplet':
                    self.aux_loss_fns['triplet'] = TripletLoss(margin=triplet_margin, mining=triplet_mining)
                    print(f"    - Triplet Margin: {triplet_margin}, Mining: {triplet_mining}")
                elif loss_name == 'center':
                    # Ensure embed_dim is provided correctly
                    self.center_loss_instance = CenterLoss(num_classes, embed_dim, device=device)
                    self.aux_loss_fns['center'] = self.center_loss_instance # Keep instance ref for param access
                elif loss_name == 'arcface':
                    # 读取 ArcFace 相关参数，并增加 Focal Loss 参数
                    arc_s = kwargs.get('arcface_s', 64.0)
                    arc_m = kwargs.get('arcface_m', 0.50)
                    arc_use_focal = kwargs.get('arcface_use_focal', False) # 新增: 是否使用 Focal
                    arc_focal_gamma = kwargs.get('arcface_focal_gamma', 2.0) # 新增: Focal gamma
                    arc_focal_alpha = kwargs.get('arcface_focal_alpha', 0.25) # 新增: Focal alpha
                    
                    self.aux_loss_fns['arcface'] = ArcFaceLoss(
                        embed_dim, num_classes, 
                        s=arc_s, 
                        m=arc_m, 
                        use_focal_loss=arc_use_focal, # 传递参数
                        focal_gamma=arc_focal_gamma,  # 传递参数
                        focal_alpha=arc_focal_alpha   # 传递参数
                    ).to(device)
                    print(f"    - ArcFace s: {arc_s}, m: {arc_m}")
                    # 在打印信息中也包含 Focal Loss 的状态
                    if arc_use_focal:
                        print(f"      -> Using Focal Loss (gamma={arc_focal_gamma}, alpha={arc_focal_alpha})")
                elif loss_name == 'cosface':
                     cos_s = kwargs.get('cosface_s', 64.0)
                     cos_m = kwargs.get('cosface_m', 0.35)
                     self.aux_loss_fns['cosface'] = CosFaceLoss(embed_dim, num_classes, s=cos_s, m=cos_m).to(device)
                     print(f"    - CosFace s: {cos_s}, m: {cos_m}")
                elif loss_name == 'cosine_sim': # Aux Cosine Similarity
                     self.aux_loss_fns['cosine_sim'] = CosineSimilarityLoss()
                elif loss_name == 'intermediate':
                     # 为中间层特征创建辅助损失函数
                     self.aux_loss_fns['intermediate'] = IntermediateLayerLoss(
                         embed_dim, num_classes, 
                         loss_type=intermediate_loss_type,
                         temperature=intermediate_temp,
                         device=device
                     )
                     print(f"    - Intermediate Layer Loss Type: {intermediate_loss_type}, Temp: {intermediate_temp}")
                else:
                    print(f"Warning: Unknown auxiliary loss type '{loss_name}' specified. Skipping.")
        
        # 初始化中间层特征损失字典
        self.intermediate_layer_losses = {}

    def forward(self, model_output, labels):
        """
        Args:
            model_output: Output from the model.
                          - If main='infonce' or 'cosine_sim': Expects (image_features [N,D], prompt_features [C,D] or [N,D] if pre-selected)
                                                              或 (image_features [N,D], prompt_features [C,D], diversity_loss)
                          - If main='ce': Expects logits [N,C]
                          - If main='ce_infonce': Expects (logits [N,C], image_features [N,D], prompt_features [C,D])
                                                或 (logits [N,C], image_features [N,D], prompt_features [C,D], diversity_loss)
                          
                          如果启用了中间层辅助损失，model_output的最后一个元素应该是aux_loss_features字典
            labels (Tensor): Ground truth labels (N).
        """
        total_loss = 0.0
        image_features = None # Initialize
        prompt_features = None # Initialize
        aux_loss_features = None # 初始化中间层特征字典
        diversity_loss = None # 初始化多样性损失
        
        # 检查是否包含中间层特征或多样性损失
        if isinstance(model_output, tuple) and len(model_output) > 0:
            # 先检查并处理可能的多样性损失
            if isinstance(model_output[-1], torch.Tensor) and model_output[-1].dim() == 0:
                # 如果最后一个元素是标量张量，可能是多样性损失
                diversity_loss = model_output[-1]
                model_output = model_output[:-1]
                if len(model_output) == 1:
                    model_output = model_output[0]
                
            # 然后检查并处理可能的中间层特征
            if isinstance(model_output, tuple) and len(model_output) > 0 and isinstance(model_output[-1], dict):
                if 'intermediate' not in self.aux_losses_config:
                    # 如果最后一个元素是字典但没有启用中间层损失，去掉它
                    model_output = model_output[:-1]
                    if len(model_output) == 1:
                        model_output = model_output[0]
                else:
                    # 如果启用了中间层损失，提取中间层特征
                    aux_loss_features = model_output[-1]
                    model_output = model_output[:-1]
                    if len(model_output) == 1:
                        model_output = model_output[0]

        # --- Calculate Main Loss ---
        if self.main_loss_type == 'infonce':
            if not isinstance(model_output, tuple) or len(model_output) < 2:
                raise TypeError("Expected model_output=(image_features, prompt_features) for main_loss_type='infonce'")
            image_features, prompt_features = model_output[:2] # prompt_features可以是 (C, D) 或 (C, P, D) 格式
            # 存储prompt_features以供辅助损失使用
            self.prompt_features = prompt_features
            
            # 检查特征维度匹配
            if prompt_features.dim() == 2 and image_features.shape[1] != prompt_features.shape[1]:
                raise ValueError(f"Embedding dimension mismatch: image D={image_features.shape[1]}, prompt D={prompt_features.shape[1]}")
            elif prompt_features.dim() == 3 and image_features.shape[1] != prompt_features.shape[2]:
                raise ValueError(f"Embedding dimension mismatch: image D={image_features.shape[1]}, prompt D={prompt_features.shape[2]}")
            
            main_loss = self.main_loss_fn(image_features, prompt_features, labels.long())
            total_loss += main_loss
            
            # 如果有多样性损失，添加到总损失中
            if diversity_loss is not None:
                total_loss += diversity_loss
                if torch.rand(1).item() < 0.05:  # 5%概率打印，避免日志过多
                    print(f"添加多样性损失: {diversity_loss.item():.4f}")

        elif self.main_loss_type == 'cosine_sim':
            if not isinstance(model_output, tuple) or len(model_output) < 2:
                raise TypeError("Expected model_output=(image_features, prompt_features) for main_loss_type='cosine_sim'")
            image_features, prompt_features = model_output[:2] # prompt_features expected shape (C, D)
            # Validate prompt_features shape
            if prompt_features.dim() != 2:
                raise ValueError(f"Cosine Similarity main loss requires prompt_features shape (C, D), got {prompt_features.shape}")
            if image_features.shape[1] != prompt_features.shape[1]:
                 raise ValueError(f"Embedding dimension mismatch: image D={image_features.shape[1]}, prompt D={prompt_features.shape[1]}")

            # Select positive prompt features based on labels
            positive_prompt_features = prompt_features[labels.long()] # Shape (N, D)
            main_loss = self.main_loss_fn(image_features, positive_prompt_features)
            total_loss += main_loss
            
            # 如果有多样性损失，添加到总损失中
            if diversity_loss is not None:
                total_loss += diversity_loss
                if torch.rand(1).item() < 0.05:  # 5%概率打印，避免日志过多
                    print(f"添加多样性损失: {diversity_loss.item():.4f}")

        elif self.main_loss_type == 'ce':
             # Assume model_output contains logits
             if isinstance(model_output, tuple): # Handle case where model might still return tuple
                  print("Warning: Received tuple output for main_loss_type='ce', using first element as logits.")
                  logits = model_output[0]
             else:
                  logits = model_output
             main_loss = self.main_loss_fn(logits, labels.long())
             total_loss += main_loss
             # No aux losses allowed here
             
        elif self.main_loss_type == 'ce_infonce':
            # 处理CE+InfoNCE组合损失
            if not isinstance(model_output, tuple) or len(model_output) < 3:
                raise TypeError("Expected model_output=(logits, image_features, prompt_features) for main_loss_type='ce_infonce'")
                
            logits, image_features, prompt_features = model_output[:3]
            
            # 存储特征以供辅助损失使用
            self.prompt_features = prompt_features
            
            # 检查特征维度匹配
            if prompt_features.dim() == 2 and image_features.shape[1] != prompt_features.shape[1]:
                raise ValueError(f"Embedding dimension mismatch: image D={image_features.shape[1]}, prompt D={prompt_features.shape[1]}")
            elif prompt_features.dim() == 3 and image_features.shape[1] != prompt_features.shape[2]:
                raise ValueError(f"Embedding dimension mismatch: image D={image_features.shape[1]}, prompt D={prompt_features.shape[2]}")
            
            # 计算交叉熵损失
            ce_loss = self.ce_loss_fn(logits, labels.long())
            
            # 计算InfoNCE损失
            infonce_loss = self.infonce_loss_fn(image_features, prompt_features, labels.long())
            
            # 线性加权组合
            main_loss = ce_loss + self.contrast_weight * infonce_loss
            
            # 如果有多样性损失，添加到总损失中
            if diversity_loss is not None:
                total_loss += diversity_loss
                if torch.rand(1).item() < 0.05:  # 5%概率打印，避免日志过多
                    print(f"添加多样性损失: {diversity_loss.item():.4f}")
            
            # 输出各部分损失值，方便调试
            if torch.distributed.get_rank() == 0 and torch.rand(1).item() < 0.1:  # 10%概率打印，避免日志过多
                loss_info = f"CE损失: {ce_loss.item():.4f}, InfoNCE损失: {infonce_loss.item():.4f}, 权重: {self.contrast_weight}"
                if diversity_loss is not None:
                    loss_info += f", 多样性损失: {diversity_loss.item():.4f}"
                loss_info += f", 组合损失: {main_loss.item():.4f}"
                print(loss_info)
            
            total_loss += main_loss

        else:
             # This case should not be reached due to __init__ check
             raise ValueError(f"Internal Error: Calculation logic missing for main_loss_type: {self.main_loss_type}")

        # --- Calculate Auxiliary Losses (Only if main_loss_type was 'infonce' or 'ce_infonce') ---
        # This block is only entered if self.aux_loss_fns is populated,
        # which only happens if main_loss_type was valid and aux_losses were specified.
        if self.aux_loss_fns:
            # Ensure we have image_features
            if image_features is None:
                 # This should not happen if main_loss_type was 'infonce' or 'ce_infonce'
                 raise RuntimeError("Internal Error: image_features are missing for auxiliary loss calculation")

            for loss_name, loss_fn in self.aux_loss_fns.items():
                weight = self.aux_weights.get(loss_name, 1.0)
                if weight == 0: continue # Skip if weight is zero

                try:
                    if loss_name == 'triplet':
                        # 检查是否使用batch hard mining
                        if self.triplet_mining in ['hard', 'semi-hard'] and image_features.size(0) > 1:
                            # 直接使用batch hard mining (不需要构建triplets)
                            aux_loss = loss_fn.batch_hard_mining(image_features, labels)
                        else:
                            # Fall back to random triplet selection inside batch
                            if not self.triplet_fallback_logged: # Log this only once
                                print(f"Warning: Using fallback random triplet selection for triplet loss.")
                                self.triplet_fallback_logged = True

                            # Check if we have at least one sample per class for effective triplet mining
                            # Group indices by class
                            class_indices = {}
                            for i, label in enumerate(labels.cpu().numpy()):
                                if label not in class_indices:
                                    class_indices[label] = []
                                class_indices[label].append(i)

                            # Filter to classes with at least 2 samples (anchor + positive)
                            valid_classes = [cls_idx for cls_idx, indices in class_indices.items() if len(indices) >= 2]
                            
                            if len(valid_classes) < 1 or len(class_indices) < 2:
                                # Not enough classes or samples per class for triplet loss
                                if torch.rand(1).item() < 0.01: # Only log occasionally 
                                    print(f"Warning: Skipping triplet loss - need at least 2 samples from 1 class and 1 sample from another.")
                                continue # Skip this loss
                                
                            # Pre-allocate tensors for anchor, positive, negative
                            batch_size = min(image_features.size(0), 64) # Limit triplet count for efficiency
                            anchor_idx = []
                            positive_idx = []
                            negative_idx = []
                            
                            # For each triplet:
                            for _ in range(batch_size):
                                # 1. Select random class with at least 2 samples
                                anchor_class = np.random.choice(valid_classes)
                                
                                # 2. Select anchor and positive from this class
                                a_idx, p_idx = np.random.choice(class_indices[anchor_class], 2, replace=False)
                                
                                # 3. Select negative class (different from anchor class)
                                negative_classes = [c for c in class_indices.keys() if c != anchor_class]
                                if not negative_classes: 
                                    # Should not happen given earlier check, but just in case
                                    continue
                                negative_class = np.random.choice(negative_classes)
                                
                                # 4. Select random negative sample
                                n_idx = np.random.choice(class_indices[negative_class])
                                
                                # Add to batch
                                anchor_idx.append(a_idx)
                                positive_idx.append(p_idx)
                                negative_idx.append(n_idx)

                            if not anchor_idx:  # If we couldn't build any triplets
                                continue
                                
                            # Extract features for the triplet batch
                            anchor_feats = image_features[anchor_idx]
                            positive_feats = image_features[positive_idx]
                            negative_feats = image_features[negative_idx]
                            
                            # Compute triplet loss
                            aux_loss = loss_fn(anchor_feats, positive_feats, negative_feats)

                    elif loss_name == 'intermediate' and aux_loss_features is not None:
                        # 处理中间层特征的辅助损失
                        intermediate_total_loss = 0.0
                        
                        # 遍历所有中间层特征
                        for layer_idx, intermediate_features in aux_loss_features.items():
                            # 确保特征维度正确
                            if intermediate_features.dim() != 3:  # [B, L, D]
                                # 如果是[B, D]形式，则不需要池化
                                layer_features = intermediate_features
                            else:
                                # 对序列特征进行平均池化
                                layer_features = intermediate_features.mean(dim=1)  # [B, D]
                            
                            # 计算该层的损失
                            layer_loss = loss_fn(layer_features, labels.long())
                            
                            # 记录每层的损失
                            self.intermediate_layer_losses[layer_idx] = layer_loss.item()
                            
                            # 添加到总损失
                            intermediate_total_loss += layer_loss
                        
                        # 如果有多层，取平均值
                        if len(aux_loss_features) > 0:
                            aux_loss = intermediate_total_loss / len(aux_loss_features)
                        else:
                            aux_loss = torch.tensor(0.0, device=image_features.device)
                            
                        # 打印中间层损失统计
                        if torch.distributed.get_rank() == 0 and torch.rand(1).item() < 0.1:  # 10%概率打印
                            print(f"中间层损失: {self.intermediate_layer_losses}")
                    
                    # 保留现有损失函数的处理逻辑
                    elif loss_name in ['center', 'arcface', 'cosface', 'cosine_sim']:
                        # ... 现有损失函数处理代码 ...
                        if loss_name == 'center':
                            aux_loss = loss_fn(image_features, labels)
                            # NOTE: Center loss alpha now applied outside this class. See engine.py.
                        elif loss_name == 'arcface':
                            # Send normalized features to ArcFace
                            normalized_features = F.normalize(image_features, p=2, dim=1)
                            aux_loss = loss_fn(normalized_features, labels)
                        elif loss_name == 'cosface':
                            # Send normalized features to CosFace
                            normalized_features = F.normalize(image_features, p=2, dim=1)
                            aux_loss = loss_fn(normalized_features, labels)
                        elif loss_name == 'cosine_sim':
                            # Use prompt_features if available (could be first or second element)
                            if prompt_features is not None:
                                # Get prompt features for each class (supports multi-prompt per class)
                                if prompt_features.dim() == 3: # [C, P, D]
                                    # Select one prompt per class based on label
                                    # For simplicity, use first prompt for now. Could be refined.
                                    # We would match image feature shape [B, D] with prompt shape [C, P, D]
                                    # by extracting desired prompts from each class based on some logic.
                                    prompt_to_use = prompt_features[:, 0, :] # Use first prompt per class for all
                                    class_prompt_features = prompt_to_use[labels.long()]
                                elif prompt_features.dim() == 2: # [C, D]
                                    # Simple lookup based on label
                                    class_prompt_features = prompt_features[labels.long()]
                                else:
                                    raise ValueError(f"Unsupported prompt_features dimension: {prompt_features.dim()}")
                                
                                # Calculate cosine similarity loss
                                aux_loss = loss_fn(image_features, class_prompt_features)
                            else:
                                # No prompt features available to calculate cosine similarity
                                print("Warning: No prompt_features available for cosine_sim loss. Skipping.")
                                continue
                    
                    # Apply weight
                    total_loss += weight * aux_loss
                    
                except Exception as e:
                    print(f"Error computing auxiliary loss '{loss_name}': {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with other losses instead of failing

        return total_loss
