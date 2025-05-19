import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os

from models_mamba_clip import VisionMamba
from datasets import build_dataset
from utils import load_model_weights, accuracy


def extract_features(model, dataloader, device):
    model.eval()
    all_image_features = []
    all_labels = []

    with torch.no_grad():
        for samples, targets in dataloader:
            samples = samples.to(device)
            targets = targets.to(device)

            # 返回格式: (image_features, prompt_features) 或 (logits, image_features, prompt_features)
            output = model(samples)
            if isinstance(output, tuple) and len(output) == 2:
                image_features, prompt_features = output
            elif isinstance(output, tuple) and len(output) == 3:
                _, image_features, prompt_features = output
            else:
                raise ValueError("Unsupported output format from model")

            all_image_features.append(image_features)
            all_labels.append(targets)

        all_image_features = torch.cat(all_image_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

    return all_image_features, all_labels, prompt_features


def evaluate_topk(image_features, prompt_features, labels, topk=(1, 5)):
    logits = image_features @ prompt_features.T
    accs = accuracy(logits, labels, topk=topk)
    return accs


def main(args):
    device = torch.device(args.device)

    # 构建测试集
    dataset_test = build_dataset(is_train=False, args=args)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 构建模型
    model = VisionMamba(
        img_size=args.input_size,
        patch_size=16,
        stride=args.stride,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_classes=args.num_classes,
        use_class_aware_prompts=True,  # 或者 use_class_prototype_prompts=True
    )
    model.to(device)

    # 加载模型权重
    load_model_weights(model, args.resume)

    # 提取特征
    image_features, labels, prompt_features = extract_features(model, dataloader_test, device)

    # 标准化
    image_features = F.normalize(image_features, dim=-1)
    prompt_features = F.normalize(prompt_features.squeeze(0), dim=-1)

    # Top-1 / Top-5 评估
    top1, top5 = evaluate_topk(image_features, prompt_features, labels)
    print(f"✅ Top-1 Accuracy: {top1:.2f}%, Top-5 Accuracy: {top5:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='Path to test data')
    parser.add_argument('--resume', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--embed-dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=24)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--num-classes', type=int, default=10)

    args = parser.parse_args()
    main(args)
