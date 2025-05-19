# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main_clip.py (modified for models_mamba_clip)
"""
import math
import sys
from typing import Iterable, Optional
import torch
from pathlib import Path
import torch.nn.functional as F # Add F for ArcFace evaluation and feature normalization
import torch.distributed as dist
import os

# --- Re-add Handle parent directory import ---
# Add the parent directory (project root) to the Python path
# Necessary when running script directly, esp. with launch tools
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Handle parent directory import ---

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

# Use absolute imports (relative to project root)
# import vim_clip.losses as losses # Comment out or delete
import losses
import utils # Already correct

# Add standard CrossEntropyLoss import if not implicitly available via torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# Add KMeans clustering functionality for computing sub-prototypes
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, fallback to simple prototype calculation")

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda', normalize_centers=True, normalize_features=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.normalize_centers = normalize_centers
        self.normalize_features = normalize_features
        # Use smaller initial values
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device) * 0.01)
        self.grad_accumulation_steps = 4
        self.current_step = 0
        self.accumulated_grad = None

    def forward(self, features, labels):
        """
        Calculate center loss
        Args:
            features (Tensor): Feature vectors (batch_size, feat_dim)
            labels (Tensor): Class labels (batch_size)
        """
        batch_size = features.size(0)
        
        if labels.device != features.device:
            labels = labels.to(features.device)
            
        if self.normalize_features:
            features_norm = F.normalize(features.clone(), p=2, dim=1)
        else:
            features_norm = features.clone()
            
        if self.normalize_centers:
            centers_norm = F.normalize(self.centers, p=2, dim=1)
        else:
            centers_norm = self.centers
            
        centers_batch = centers_norm.index_select(0, labels)
        
        if self.normalize_features and self.normalize_centers:
            sim = torch.sum(features_norm * centers_batch, dim=1)
            dist = 1.0 - sim
        else:
            dist = torch.sum((features_norm - centers_batch) ** 2, dim=1)
            
        loss = torch.mean(dist)
        return loss

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, center_loss=None, center_loss_weight=0.0,
                    log_writer=None, center_params=None):
    """
    Modified to work with CombinedLoss, handle CenterLoss updates,
    and manage Mixup/CoSub compatibility based on args.main_loss_type.
    """
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    using_class_aware_sampler = (hasattr(data_loader, 'batch_sampler') and 
                                hasattr(data_loader.batch_sampler, 'update_class_weights') and
                                hasattr(data_loader.batch_sampler, 'use_softmax_balance') and
                                data_loader.batch_sampler.use_softmax_balance)
    
    if using_class_aware_sampler:
        print("Using ClassAwareBatchSampler with Softmax Balance")
        
        num_classes = data_loader.batch_sampler.num_classes
        class_loss_sum = torch.zeros(num_classes, device=device)
        class_loss_count = torch.zeros(num_classes, device=device)

    # --- Check for Center Loss in criterion ---
    # Need access to the actual CombinedLoss instance, handling potential Distillation wrapping
    unwrapped_criterion = criterion
    if isinstance(criterion, losses.DistillationLoss):
        unwrapped_criterion = criterion.base_criterion

    has_center_loss = False
    center_loss_alpha = 0.0 # Default, should be set from args if used
    center_loss_instance = None
    if isinstance(unwrapped_criterion, losses.CombinedLoss) and 'center' in unwrapped_criterion.aux_loss_fns:
         has_center_loss = True
         center_loss_instance = unwrapped_criterion.aux_loss_fns['center']
         center_loss_alpha = getattr(args, 'center_alpha', 0.5) # Get alpha from args
         print(f"CenterLoss detected. Alpha for center updates: {center_loss_alpha}")

    # --- Mixup/Cutmix/CoSub Compatibility Check ---
    if args.main_loss_type != 'ce':
        if mixup_fn is not None:
            print(f"Warning: Disabling Mixup/Cutmix as main_loss_type is '{args.main_loss_type}', not 'ce'.")
            mixup_fn = None # Disable mixup
        if args.cosub:
            print(f"Warning: Disabling CoSub as main_loss_type is '{args.main_loss_type}', not 'ce'.")
            args.cosub = False # Disable CoSub
    else: # Main loss is 'ce'
        if mixup_fn is not None:
             print("Mixup/Cutmix enabled with 'ce' loss.")
        if args.cosub:
             print("CoSub enabled with 'ce' loss.")
             # CoSub requires its own criterion if main criterion is CE
             criterion_cosub = torch.nn.BCEWithLogitsLoss()
        else:
             criterion_cosub = None # Ensure it's None if cosub is false

    # Removed old loss type checks (is_contrastive_loss, is_triplet_loss, etc.)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            # Mixup handled by SoftTargetCrossEntropy or specific logic if main_loss is CE

        if args.cosub: # Only possible if main_loss_type == 'ce' and args.cosub=True
            samples = torch.cat((samples, samples), dim=0)
            # Don't duplicate targets if mixup is active, as mixup returns combined targets
            if mixup_fn is None:
                targets = torch.cat((targets, targets), dim=0)

        with amp_autocast():
            # Model forward pass
            model_output = model(samples,
                                if_random_cls_token_position=getattr(args, 'if_random_cls_token_position', False),
                                if_random_token_rank=getattr(args, 'if_random_token_rank', False))

            # --- Generic Loss Calculation using Criterion ---
            # CombinedLoss handles internal logic based on main_loss_type and aux_losses
            if not args.cosub:
                try:
                    loss = criterion(model_output, targets)
                except RuntimeError as e:
                    if "Expected all tensors to be on the same device" in str(e):
                        print(f"Caught device mismatch error: {e}")
                        print(f"Using fallback loss calculation method")
                        
                        if args.main_loss_type == 'infonce' and isinstance(model_output, tuple) and len(model_output) == 2:
                            image_features, prompt_features = model_output
                           
                            if hasattr(criterion, 'main_loss_fn') and hasattr(criterion.main_loss_fn, 'forward'):
                                loss = criterion.main_loss_fn(image_features, prompt_features, targets)
                            else:
                                loss = F.cross_entropy(image_features @ prompt_features.t() / 0.07, targets)
                                
                        elif args.main_loss_type == 'ce_infonce' and isinstance(model_output, tuple) and len(model_output) == 3:
                            logits, image_features, prompt_features = model_output
                            
                            loss = F.cross_entropy(logits, targets)
                            print("Backup method: Use only the CE loss and ignore the InfoNCE part.")
                            
                        elif args.main_loss_type == 'ce' and isinstance(model_output, torch.Tensor):
                            # Standard cross-entropy loss
                            loss = F.cross_entropy(model_output, targets)
                            
                        else:
                            # Unhandled case
                            print("The reserve loss cannot be calculated as there is no appropriate model output format.")
                            raise e
                    else:
                        # Re-raise other errors
                        raise e
            else:
                # CoSub logic (only runs if main_loss_type=='ce' and args.cosub=True)
                if not isinstance(model_output, torch.Tensor): # CoSub expects logits
                     raise TypeError(f"CoSub requires model to output logits, but got {type(model_output)}")
                logits = model_output
                logits = torch.split(logits, logits.shape[0]//2, dim=0)
                target_splits = torch.split(targets, targets.shape[0]//2, dim=0)

                # Use BCEWithLogitsLoss for CoSub components
                loss = 0.25 * criterion_cosub(logits[0], target_splits[0].float()) # Mixup targets are float
                loss = loss + 0.25 * criterion_cosub(logits[1], target_splits[1].float())
                loss = loss + 0.25 * criterion_cosub(logits[0], logits[1].detach().sigmoid())
                loss = loss + 0.25 * criterion_cosub(logits[1], logits[0].detach().sigmoid())

            # --- Store features if CenterLoss is active ---
            image_features_for_center_update = None
            if has_center_loss:
                 if isinstance(model_output, tuple):
                     if len(model_output) == 2 and args.main_loss_type == 'infonce':
                         image_features_for_center_update = model_output[0].detach()
                     elif len(model_output) == 3 and args.main_loss_type == 'ce_infonce':
                         image_features_for_center_update = model_output[1].detach()
                     elif args.main_loss_type == 'ce':
                         if len(model_output) > 1:
                             image_features_for_center_update = model_output[1].detach()
                         else:
                             print("Warning: CE main loss with tuple output but no features for CenterLoss")
                 elif args.main_loss_type == 'ce' and isinstance(unwrapped_criterion, losses.CombinedLoss):
                      print("Warning: CenterLoss with CE main loss requires features. Check model/CombinedLoss.")
                      pass # Placeholder - Requires CombinedLoss modification to store features with CE
                 else:
                     print("Warning: Cannot get features for CenterLoss update.")

        if getattr(args, 'if_nan2num', False):
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if getattr(args, 'if_continue_inf', False):
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # --- Standard Backward Pass ---
        if isinstance(loss_scaler, timm.utils.NativeScaler):
             is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
             # Scale loss, calculate gradients
             loss_scaler(loss, optimizer, clip_grad=max_norm,
                     parameters=model.parameters(), create_graph=is_second_order,
                     update_grad=False) # Don't update gradients yet, need center loss step
        else:
             loss.backward()
             # Gradients are now computed
        
        if utils.is_main_process() and i == 0:  
           
            model_unwrapped = model.module if hasattr(model, 'module') else model
            
            print(f"\n[Epoch {epoch}] Gradient statistics:")
            
            # 1. Monitor prompt gradients - simplified version
            has_prompt = False
            for name, param in model_unwrapped.named_parameters():
                if 'prompt_tokens' in name and param.requires_grad and param.grad is not None:
                    has_prompt = True
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.abs().mean().item()
                    param_norm = param.norm().item()
                    print(f"  Prompt gradient summary - Norm: {grad_norm:.6f}, Mean: {grad_mean:.6f}, Parameter norm: {param_norm:.6f}")
                    
                    # Output warning if gradient is too small
                    if grad_norm < 1e-4:
                        print(f"  Warning: Prompt gradient too small ({grad_norm:.8f})")
                    break  # Only show statistics for one prompt parameter group
            
            if not has_prompt:
                print("  No prompt parameters found")
            
            # 2. Monitor main backbone layers gradients - simplified version
            # Selectively monitor: last blocks and norm layers
            backbone_summary = {}
            
            # Collect these layer parameters
            for name, param in model_unwrapped.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if 'blocks.' in name:
                        block_idx = int(name.split('blocks.')[1].split('.')[0])
                        if block_idx == len(getattr(model_unwrapped, 'blocks', [])) - 1:
                            layer_name = 'blocks[-1]'
                            if layer_name not in backbone_summary:
                                backbone_summary[layer_name] = {'grad_norm': 0, 'param_norm': 0, 'count': 0}
                            backbone_summary[layer_name]['grad_norm'] += param.grad.norm().item() ** 2
                            backbone_summary[layer_name]['param_norm'] += param.norm().item() ** 2
                            backbone_summary[layer_name]['count'] += 1
                    elif 'norm' in name and 'blocks' not in name:
                        layer_name = 'norm'
                        if layer_name not in backbone_summary:
                            backbone_summary[layer_name] = {'grad_norm': 0, 'param_norm': 0, 'count': 0}
                        backbone_summary[layer_name]['grad_norm'] += param.grad.norm().item() ** 2
                        backbone_summary[layer_name]['param_norm'] += param.norm().item() ** 2
                        backbone_summary[layer_name]['count'] += 1
            
            # Print simplified statistics
            if backbone_summary:
                print("\n  Backbone gradient summary:")
                for layer_name, stats in backbone_summary.items():
                    if stats['count'] > 0:
                        grad_norm = (stats['grad_norm'] ** 0.5)
                        param_norm = (stats['param_norm'] ** 0.5)
                        print(f"    {layer_name}: Gradient norm: {grad_norm:.6f}, Parameter norm: {param_norm:.6f}, "
                              f"Gradient/Parameter ratio: {grad_norm/max(param_norm, 1e-6):.6f}")

        # --- Center Loss Update (if applicable) ---
        if has_center_loss and center_loss_instance is not None:
            # Update center parameters using detach to avoid gradient issues
            with torch.no_grad():
                for param in center_loss_instance.parameters():
                    if param.grad is not None:
                        # Gradient clipping to avoid gradient explosion
                        grad_norm = torch.norm(param.grad.data, p=2)
                        if grad_norm > 1.0:
                            scaled_grad = param.grad.data / grad_norm
                        else:
                            scaled_grad = param.grad.data
                        
                        # Create new parameters, avoid in-place operations
                        new_param_data = param.data - center_loss_alpha * 0.1 * scaled_grad
                        param.copy_(new_param_data)

        # --- Optimizer Step ---
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            if max_norm is not None and max_norm > 0:
                loss_scaler.unscale_(optimizer) # Unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            loss_scaler.step(optimizer) # Optimizer step
            loss_scaler.update() # Update scaler state
        else:
            # Clip gradients if needed (already computed)
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step() # Optimizer step


        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # --- Record class losses ---
        if using_class_aware_sampler and targets.dim() == 1:  # Only when targets are class indices
            # Collect classes in the current batch
            batch_classes = targets.detach()
            # Update class loss statistics
            for i, cls_idx in enumerate(batch_classes):
                if cls_idx < num_classes:  # Prevent class index out of bounds
                    class_loss_sum[cls_idx] += loss_value / len(targets)  # Average across samples
                    class_loss_count[cls_idx] += 1

        # Reset first_batch flag after processing each batch
        first_batch = False

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # --- Update ClassAwareBatchSampler's class weights ---
    if using_class_aware_sampler:
        print("Updating ClassAwareBatchSampler class weights")
        # Calculate average loss for each class
        class_loss_avg = class_loss_sum.clone()
        for i in range(num_classes):
            if class_loss_count[i] > 0:
                class_loss_avg[i] = class_loss_sum[i] / class_loss_count[i]
            else:
                # For classes with no samples, use the average loss
                class_loss_avg[i] = class_loss_sum.sum() / max(class_loss_count.sum(), 1)
        
        # Convert loss values to numpy arrays
        class_loss_avg_np = class_loss_avg.detach().cpu().numpy()
        
        # Update sampler weights
        data_loader.batch_sampler.update_class_weights(class_loss_avg_np)
        
        # Record class loss statistics
        print("Class loss statistics:")
        for i in range(num_classes):
            if class_loss_count[i] > 0:
                print(f"  Class {i}: Average loss = {class_loss_avg[i]:.4f}, Sample count = {class_loss_count[i]}")

    # Return results
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Replace the original compute_entropy function
def compute_entropy(probs):
    """
    Compute the entropy of softmax outputs
    Args:
        probs: softmax probability vector, shape [N, C]
    Returns:
        entropy vector, shape [N]
    """
    # Add a small value to avoid log(0)
    eps = 1e-8
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def compute_uncertainty_score(probs, logits=None):
    """
    Compute multi-dimensional uncertainty scores, combining multiple metrics
    
    Args:
        probs: softmax probability vector, shape [N, C]
        logits: original logits, shape [N, C]
    Returns:
        uncertainty_dict: dictionary containing multiple uncertainty metrics
    """
    # Compute entropy
    eps = 1e-8
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)
    
    # Compute top1 and top2 probability margin
    topk_probs, _ = torch.topk(probs, k=min(3, probs.size(1)), dim=1)
    top1_probs = topk_probs[:, 0]
    top2_probs = topk_probs[:, 1]
    top3_probs = topk_probs[:, 2] if probs.size(1) >= 3 else topk_probs[:, 0]
    prob_margin_12 = top1_probs - top2_probs
    prob_margin_13 = top1_probs - top3_probs
    
    # Compute logits margin
    logits_margin = None
    if logits is not None:
        topk_logits, _ = torch.topk(logits, k=min(3, logits.size(1)), dim=1)
        logits_margin = topk_logits[:, 0] - topk_logits[:, 1]
    
    # Normalize entropy (range 0-1)
    max_possible_entropy = torch.log(torch.tensor(probs.size(1), dtype=torch.float, device=probs.device))
    normalized_entropy = entropy / max_possible_entropy
    
    # Combine uncertainty metrics: higher entropy, smaller margin, higher uncertainty
    uncertainty_margin = normalized_entropy * (1.0 - prob_margin_12)
    # Add quadratic term for more sensitivity
    uncertainty_quad = normalized_entropy * (1.0 - prob_margin_12**2)
    
    return {
        'entropy': entropy,
        'norm_entropy': normalized_entropy,
        'prob_margin_12': prob_margin_12,
        'prob_margin_13': prob_margin_13,
        'logits_margin': logits_margin,
        'uncertainty_margin': uncertainty_margin,
        'uncertainty_quad': uncertainty_quad
    }

# Replace the original class prototype computation function
def compute_class_prototypes(image_features, labels, logits, num_classes):
    """
    Compute weighted class prototypes to reduce the impact of low-quality samples
    
    Args:
        image_features: image features, shape [N, D]
        labels: sample labels, shape [N]
        logits: model output logits, shape [N, C]
        num_classes: total number of classes
    Returns:
        class prototypes, shape [num_classes, D]
    """
    device = image_features.device
    feat_dim = image_features.shape[1]
    
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=1)
    
    # Initialize weighted class prototypes
    prototypes = torch.zeros((num_classes, feat_dim), device=device)
    class_weights = torch.zeros(num_classes, device=device)
    
    # Compute each sample's weighted contribution to its true class
    for i, label in enumerate(labels):
        if label < num_classes:  # Safety check
            # Use the sample's predicted probability for its true class as weight
            # Square the predicted probability to enhance the impact of high-confidence samples
            weight = probs[i, label.long()] ** 2
            prototypes[label.long()] += image_features[i] * weight
            class_weights[label.long()] += weight
    
    # Compute weighted average for each class, handle cases with zero weight
    for c in range(num_classes):
        if class_weights[c] > 0:
            prototypes[c] /= class_weights[c]
        else:
            # If the class has no samples or all samples have zero weight, initialize randomly
            prototypes[c] = torch.randn_like(prototypes[c]) * 0.01
    
    # L2 normalize prototypes
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    return prototypes

# Add a function to identify hard-to-distinguish class pairs
def get_confusion_pairs(confusion_matrix, threshold=0.15):
    """
    Identify easily confused class pairs in the confusion matrix
    
    Args:
        confusion_matrix: confusion matrix, shape [C, C]
        threshold: confusion rate threshold, pairs exceeding this are considered easily confused
    
    Returns:
        confusion_pairs: dictionary containing easily confused class pairs and their confusion rates
    """
    num_classes = confusion_matrix.shape[0]
    class_counts = confusion_matrix.sum(dim=1)
    
    confusion_pairs = {}
    highly_confused_pairs = []
    
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            # Compute the rate of i being predicted as j + j being predicted as i, divided by 2
            if class_counts[i] > 0 and class_counts[j] > 0:
                i_to_j_rate = confusion_matrix[i, j].float() / class_counts[i]
                j_to_i_rate = confusion_matrix[j, i].float() / class_counts[j]
                avg_confusion = (i_to_j_rate + j_to_i_rate) / 2
                
                if avg_confusion > threshold:
                    confusion_pairs[(i, j)] = {
                        'rate': avg_confusion.item(),
                        'i_to_j': i_to_j_rate.item(),
                        'j_to_i': j_to_i_rate.item()
                    }
                    
                    if avg_confusion > threshold * 1.5:  # Highly confused class pairs
                        highly_confused_pairs.append((i, j, avg_confusion.item()))
    
    # Sort by confusion rate
    highly_confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return confusion_pairs, highly_confused_pairs

# Use K-means clustering to compute subclass prototypes
def compute_subclass_prototypes(image_features, labels, num_classes, num_clusters=3, min_samples=10):
    """
    Compute multiple subclass prototypes for each class to better capture intra-class variation
    
    Args:
        image_features: image features, shape [N, D]
        labels: sample labels, shape [N]
        num_classes: total number of classes
        num_clusters: maximum number of clusters per class
        min_samples: minimum number of samples required per cluster
    
    Returns:
        class_subprototypes: multiple subclass prototypes for each class
    """
    device = image_features.device
    feat_dim = image_features.shape[1]
    labels_np = labels.cpu().numpy()
    
    # Check if sklearn is available
    if not SKLEARN_AVAILABLE:
        # Fallback to regular prototype computation
        prototypes = torch.zeros((num_classes, 1, feat_dim), device=device)
        counts = torch.zeros(num_classes, device=device)
        
        # Compute average features for each class
        for i, label in enumerate(labels):
            if label < num_classes:
                prototypes[label, 0] += image_features[i]
                counts[label] += 1
        
        # Compute averages
        for c in range(num_classes):
            if counts[c] > 0:
                prototypes[c, 0] /= counts[c]
        
        # Normalize
        prototypes = F.normalize(prototypes, p=2, dim=2)
        return prototypes, None
    
    # Use K-means to compute subclass prototypes
    image_features_np = image_features.cpu().numpy()
    
    # Storage for subclass prototypes for each class
    class_subprototypes = torch.zeros((num_classes, num_clusters, feat_dim), device=device)
    subproto_counts = torch.zeros((num_classes, num_clusters), device=device)
    cluster_info = {}
    
    for c in range(num_classes):
        # Find samples of this class
        class_indices = np.where(labels_np == c)[0]
        
        if len(class_indices) < min_samples:
            # Too few samples, use class average as the only prototype
            class_features = image_features_np[class_indices]
            class_prototype = np.mean(class_features, axis=0)
            class_subprototypes[c, 0] = torch.tensor(class_prototype, device=device)
            subproto_counts[c, 0] = len(class_indices)
            cluster_info[c] = {'n_clusters': 1, 'counts': [len(class_indices)]}
            continue
        
        # Determine number of clusters
        actual_clusters = min(num_clusters, len(class_indices) // min_samples)
        actual_clusters = max(1, actual_clusters)  # At least 1 cluster
        
        # Perform clustering
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        class_features = image_features_np[class_indices]
        cluster_labels = kmeans.fit_predict(class_features)
        
        # Compute prototype for each cluster
        cluster_counts = []
        for k in range(actual_clusters):
            cluster_indices = np.where(cluster_labels == k)[0]
            if len(cluster_indices) > 0:
                cluster_proto = np.mean(class_features[cluster_indices], axis=0)
                class_subprototypes[c, k] = torch.tensor(cluster_proto, device=device)
                subproto_counts[c, k] = len(cluster_indices)
                cluster_counts.append(len(cluster_indices))
            else:
                # Empty cluster, fill with zeros
                subproto_counts[c, k] = 0
                cluster_counts.append(0)
        
        cluster_info[c] = {'n_clusters': actual_clusters, 'counts': cluster_counts}
    
    # Normalize subclass prototypes
    for c in range(num_classes):
        for k in range(num_clusters):
            if subproto_counts[c, k] > 0:
                class_subprototypes[c, k] = F.normalize(class_subprototypes[c, k], p=2, dim=0)
    
    return class_subprototypes, cluster_info

@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, criterion=None, args=None):
    """
    Modified to use the provided criterion (CombinedLoss) and calculate
    accuracy based on args.main_loss_type determining model output.
    """
    if args is None:
         raise ValueError("`args` must be provided to evaluate function.")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    
    # Initialize statistics for each class
    num_classes = args.nb_classes
    per_class_correct = torch.zeros(num_classes, dtype=torch.int64, device=device)
    per_class_total = torch.zeros(num_classes, dtype=torch.int64, device=device)
    
    # Add confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    
    # Add new evaluation metrics
    prompt_top1_correct = 0
    prompt_top1_total = 0
    intra_class_sims = torch.zeros(num_classes, dtype=torch.float, device=device)
    intra_class_counts = torch.zeros(num_classes, dtype=torch.int64, device=device)
    inter_class_max_sims = torch.zeros((num_classes, num_classes), dtype=torch.float, device=device) - 1.0
    
    # Collect image features and labels from all batches for global metrics
    all_image_features = []
    all_prompt_features = None
    all_labels = []
    all_logits = []  # Collect logits for rescoring
    
    # New: Statistics for rescoring results
    total_samples = 0
    rescored_samples = 0
    initial_correct = 0
    rescored_correct = 0
    improved_by_rescoring = 0
    worsened_by_rescoring = 0

    # Flag variable, only print logits for the first batch
    first_batch = True
    
    # Collect class distribution for each batch
    batch_class_counts = []
    
    # Get prompts_per_class parameter from model or args
    prompts_per_class = getattr(args, 'prompts_per_class', 1)  # First try to get from args
    if prompts_per_class == 1:
        # If not in args or default value, try to get from model
        model_unwrapped = model.module if hasattr(model, 'module') else model
        if hasattr(model_unwrapped, 'prompts_per_class'):
            prompts_per_class = model_unwrapped.prompts_per_class
    
    print(f"Evaluation using prompts_per_class: {prompts_per_class}")

    # Build metrics dictionary to store all evaluation results
    metrics = {}
    
    # Add softmax entropy rescoring parameters
    entropy_threshold = getattr(args, 'entropy_threshold', 0.5)  # Entropy threshold, default 0.5, can be passed via args
    topk_for_rescoring = getattr(args, 'topk_for_rescoring', 5)  # Top-k classes considered for rescoring, default 5
    use_entropy_rescoring = getattr(args, 'use_entropy_rescoring', True)  # Whether to enable entropy rescoring, default enabled
    
    for i, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Record class distribution in the batch
        batch_classes, batch_counts = torch.unique(target, return_counts=True)
        batch_distribution = {cls_idx.item(): count.item() for cls_idx, count in zip(batch_classes, batch_counts)}
        batch_class_counts.append(batch_distribution)

        # Use amp_autocast context to adapt to mixed precision used by the model
        with amp_autocast():
            # Model forward pass
            outputs = model(images)
            
            # Handle diversity loss and auxiliary loss features
            if isinstance(outputs, tuple) and len(outputs) > 2:
                # Check if diversity loss is included (last element is scalar tensor)
                if isinstance(outputs[-1], torch.Tensor) and outputs[-1].dim() == 0:
                    # Remove diversity loss, not needed during evaluation
                    outputs = outputs[:-1]
                
                # Check if auxiliary loss features are included (last element is dictionary)
                if isinstance(outputs[-1], dict):
                    # Remove auxiliary loss features
                    outputs = outputs[:-1]
                
                # Unpack tuple if only one element remains
                if len(outputs) == 1:
                    outputs = outputs[0]
            
            # Process output based on loss type
            if args.main_loss_type == 'infonce' or args.main_loss_type == 'cosine_sim':
                # Contrastive loss expects image features and prompt features
                if not isinstance(outputs, tuple) or len(outputs) < 2:
                    raise ValueError(f"Expected tuple output for {args.main_loss_type}, got {type(outputs)}")
                
                # Extract image features and prompt features
                image_features, prompt_features = outputs[:2]
                
                # Compute loss (if criterion is provided)
                if criterion is not None:
                    loss = criterion(outputs, target)
                
                # Collect evaluation features
                all_image_features.append(image_features.cpu())
                all_labels.append(target.cpu())
                
                if all_prompt_features is None:
                    # Set prompt features for the first time
                    all_prompt_features = prompt_features.detach().cpu()
                    
                # Compute similarity between image features and prompt features
                # Ensure features are normalized
                image_features_norm = F.normalize(image_features, p=2, dim=1)
                
                # Handle multiple prompts per class
                if prompt_features.dim() == 3:  # [C, P, D]
                    # Default dimension handling
                    prompt_features_dim2 = prompt_features.clone()
                    
                    # Compute similarity between each class's multiple prompts and image features
                    batch_size = image_features.size(0)
                    logits_list = []
                    
                    # For each class
                    for c in range(prompt_features.size(0)):
                        # Get all prompts for the class
                        class_prompts = prompt_features[c]  # [P, D]
                        class_prompts_norm = F.normalize(class_prompts, p=2, dim=1)
                        
                        # Compute similarity between each image and all prompts of the class
                        # Result shape: [B, P]
                        class_sims = torch.matmul(image_features_norm, class_prompts_norm.t())
                        
                        # Take the maximum similarity for each image and the class's prompts
                        # Result shape: [B]
                        class_max_sims, _ = class_sims.max(dim=1)
                        
                        # Add to class similarity list
                        logits_list.append(class_max_sims.unsqueeze(1))
                    
                    # Concatenate maximum similarities for all classes into logits matrix
                    # Result shape: [B, C]
                    logits = torch.cat(logits_list, dim=1)
                    
                else:  # Single prompt case [C, D]
                    prompt_features_norm = F.normalize(prompt_features, p=2, dim=1)
                    # Compute similarity matrix [B, C]
                    logits = torch.matmul(image_features_norm, prompt_features_norm.t())
                    
                # Collect logits for all batches for rescoring
                all_logits.append(logits.cpu())

            elif args.main_loss_type == 'ce':
                # Cross-entropy loss expects logits
                if isinstance(outputs, tuple):
                    print(f"Warning: CE loss with tuple output, using first element as logits.")
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Compute loss
                if criterion is not None:
                    loss = criterion(logits, target)
                
                # Collect logits for all batches for rescoring
                all_logits.append(logits.cpu())
                
            elif args.main_loss_type == 'ce_infonce':
                # CE+InfoNCE combined loss
                if not isinstance(outputs, tuple) or len(outputs) < 3:
                    raise ValueError(f"Expected (logits, img_feat, prompt_feat) for ce_infonce, got {type(outputs)}")
                
                logits, image_features, prompt_features = outputs[:3]
                
                # Compute loss
                if criterion is not None:
                    loss = criterion(outputs, target)
                
                # Collect evaluation features
                all_image_features.append(image_features.cpu())
                all_labels.append(target.cpu())
                
                if all_prompt_features is None:
                    all_prompt_features = prompt_features.detach().cpu()
                
                # Collect logits for all batches for rescoring
                all_logits.append(logits.cpu())
                
            else:
                raise ValueError(f"Unsupported main_loss_type: {args.main_loss_type}")
            
            # Update loss
            if criterion is not None:
                metric_logger.update(loss=loss.item())
            
            # Compute accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, min(5, logits.shape[1])))
            batch_size = images.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            if logits.shape[1] >= 5:
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            
            # Update confusion matrix and per-class accuracy
            preds = torch.argmax(logits, dim=1)
            for t, p in zip(target.reshape(-1), preds.reshape(-1)):
                if t < num_classes:  # Safety check
                    confusion_matrix[t.long(), p.long()] += 1
                    per_class_total[t.long()] += 1
                    if t == p:
                        per_class_correct[t.long()] += 1
            
            # If first batch, print detailed information
            if i == 0 and first_batch:
                print(f"\nTest batch example - batch_size={batch_size}")
                # Limit number of samples
                print_samples = min(5, batch_size)
                if args.main_loss_type in ['infonce', 'cosine_sim', 'ce_infonce']:
                    # Print feature information
                    print(f"Image features: shape={image_features.shape}, mean={image_features.mean().item():.4f}, std={image_features.std().item():.4f}")
                    if prompt_features.dim() == 2:
                        print(f"Prompt features: shape={prompt_features.shape}, mean={prompt_features.mean().item():.4f}, std={prompt_features.std().item():.4f}")
                    elif prompt_features.dim() == 3:
                        print(f"Prompt features: shape={prompt_features.shape}, mean={prompt_features.mean().item():.4f}, std={prompt_features.std().item():.4f}")
                        
                # Print logits example
                print("\nLogits example (first 5 samples):")
                for j in range(print_samples):
                    true_label = target[j].item()
                    pred_label = preds[j].item()
                    
                    # Get top 5 predictions and scores
                    top5_values, top5_indices = torch.topk(logits[j], min(5, logits.shape[1]))
                    top5_info = ""
                    for k in range(len(top5_indices)):
                        top5_info += f"Class {top5_indices[k].item()}: {top5_values[k].item():.4f}, "
                    
                    print(f"Sample {j} - True class: {true_label}, Predicted class: {pred_label}, Top5: {top5_info}")
                    
                first_batch = False
                
                # If entropy rescoring is enabled, print entropy information
                if use_entropy_rescoring:
                    # Compute softmax probabilities
                    probs = F.softmax(logits, dim=1)
                    # Compute entropy
                    entropy = compute_entropy(probs)
                    print("\nSoftmax entropy example (first 5 samples):")
                    for j in range(print_samples):
                        print(f"Sample {j} - Entropy: {entropy[j].item():.4f}")

    # Compute per-class accuracy
    per_class_acc = torch.zeros(num_classes, dtype=torch.float, device=device)
    for i in range(num_classes):
        if per_class_total[i] > 0:
            per_class_acc[i] = per_class_correct[i] / per_class_total[i]
    
    # Summarize results
    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # Add per-class accuracy - convert to Python list
    metrics['per_class_acc'] = per_class_acc.cpu().numpy().tolist()
    metrics['per_class_total'] = per_class_total.cpu().numpy().tolist()
    
    # Compute balanced accuracy (arithmetic mean)
    class_count = (per_class_total > 0).sum().item()
    if class_count > 0:
        metrics['balanced_acc'] = per_class_acc[per_class_total > 0].mean().item()
    
    # Convert confusion matrix to numpy array
    confusion_mat_np = confusion_matrix.cpu().numpy()
    
    # Print evaluation results
    print(f"* Acc@1 {metrics['acc1']:.3f} Acc@5 {metrics.get('acc5', 0):.3f}")
    if 'balanced_acc' in metrics:
        print(f"* Balanced Acc {metrics['balanced_acc']:.3f} (average per-class accuracy)")
    
    # Print per-class sample count and accuracy
    print("\nPer-class sample count and accuracy:")
    for i in range(num_classes):
        if per_class_total[i] > 0:
            print(f"Class {i}: Sample count={per_class_total[i].item()}, Accuracy={per_class_acc[i].item():.4f}")
    
    # Output confusion matrix statistics
    print("\nConfusion matrix statistics:")
    # Identify most easily confused class pairs
    if num_classes <= 100:  # Only analyze in detail if number of classes is not too large
        num_pairs = min(10, num_classes * (num_classes - 1) // 2)  # Show at most 10 pairs
        confusion_pairs = []
        
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                if per_class_total[i] > 0 and per_class_total[j] > 0:
                    # Compute rate of i being predicted as j + j being predicted as i
                    confusion_rate = (confusion_matrix[i, j] / per_class_total[i] + 
                                     confusion_matrix[j, i] / per_class_total[j]).item() / 2
                    confusion_pairs.append((i, j, confusion_rate))
        
        # Sort by confusion rate
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Top {num_pairs} most easily confused class pairs:")
        for i, j, rate in confusion_pairs[:num_pairs]:
            print(f"Class {i} and Class {j}: Average confusion rate={rate:.4f}, "
                 f"{i}->{j} rate={confusion_matrix[i, j].item()/max(per_class_total[i].item(), 1):.4f}, "
                 f"{j}->{i} rate={confusion_matrix[j, i].item()/max(per_class_total[j].item(), 1):.4f}")
    
    # Evaluate entropy-based embedding distance rescoring
    if use_entropy_rescoring and len(all_logits) > 0 and len(all_image_features) > 0:
        print("\n===== Advanced multi-dimensional uncertainty-based intelligent embedding distance rescoring evaluation =====")
        
        # Merge all batch data, ensure all data are tensors
        if not isinstance(all_logits, torch.Tensor):
            all_logits = torch.cat(all_logits, dim=0)
        if not isinstance(all_image_features, torch.Tensor):
            all_image_features_tensor = torch.cat(all_image_features, dim=0)
        else:
            all_image_features_tensor = all_image_features
        if not isinstance(all_labels, torch.Tensor):
            all_labels_tensor = torch.cat(all_labels, dim=0)
        else:
            all_labels_tensor = all_labels
        
        # Compute softmax probabilities
        probs = F.softmax(all_logits, dim=1)
        
        # Compute confusion matrix to identify easily confused class pairs
        confusion_mat = torch.zeros((num_classes, num_classes), device=device)
        batch_size = 1000  # Process in batches to avoid memory overflow
        for i in range(0, len(all_labels_tensor), batch_size):
            end_idx = min(i + batch_size, len(all_labels_tensor))
            batch_labels = all_labels_tensor[i:end_idx]
            batch_preds = torch.argmax(all_logits[i:end_idx], dim=1)
            
            for t, p in zip(batch_labels, batch_preds):
                if t < num_classes and p < num_classes:
                    confusion_mat[t.long(), p.long()] += 1
        
        # Identify easily confused class pairs
        confusion_pairs, highly_confused_pairs = get_confusion_pairs(confusion_mat, threshold=0.15)
        
        # Print confused class pair information
        print("Identified highly confused class pairs:")
        for i, j, rate in highly_confused_pairs[:5]:  # Show only top 5
            print(f"Class {i} and Class {j}: Confusion rate={rate:.4f}")
        
        # Compute multi-dimensional uncertainty
        uncertainty_dict = compute_uncertainty_score(probs, all_logits)
        entropy = uncertainty_dict['entropy']
        normalized_entropy = uncertainty_dict['norm_entropy']
        uncertainty_scores = uncertainty_dict['uncertainty_quad']  # Use more sensitive metric
        
        # Statistics of uncertainty distribution
        uncertainty_mean = uncertainty_scores.mean().item()
        uncertainty_std = uncertainty_scores.std().item()
        uncertainty_median = uncertainty_scores.median().item()
        uncertainty_75percentile = torch.quantile(uncertainty_scores, 0.75).item()
        uncertainty_90percentile = torch.quantile(uncertainty_scores, 0.90).item()
        entropy_mean = entropy.mean().item()
        entropy_std = entropy.std().item()
        
        # Output statistics
        print(f"Uncertainty statistics: Mean={uncertainty_mean:.4f}, Std={uncertainty_std:.4f}, Median={uncertainty_median:.4f}")
        print(f"Percentiles: 75%={uncertainty_75percentile:.4f}, 90%={uncertainty_90percentile:.4f}")
        print(f"Entropy statistics: Mean={entropy_mean:.4f}, Std={entropy_std:.4f}")
        
        # Use more aggressive settings: significantly lower uncertainty threshold to rescore more samples
        # Try to read from environment variable
        alpha_env = os.environ.get('RESCORING_ALPHA')
        if alpha_env is not None:
            try:
                alpha = float(alpha_env)
                print(f"Alpha value read from environment variable: {alpha}")
            except (ValueError, TypeError):
                alpha = 0.1  # More aggressive: reduce original prediction weight
                print(f"Invalid Alpha value from environment variable, using default: {alpha}")
        else:
            alpha = 0.1  # More aggressive: reduce original prediction weight
        
        # Get threshold settings
        if entropy_threshold > 0:
            print(f"Using user-specified entropy threshold: {entropy_threshold}")
            uncertainty_threshold = entropy_threshold
        else:
            # More aggressive: use lower percentile as threshold to rescore more samples
            uncertainty_threshold = uncertainty_75percentile * 0.9  # Use 90% of 75th percentile
            print(f"Using adaptive uncertainty threshold (lowered): {uncertainty_threshold:.4f}")
        
        min_improvement = 0.01  # More aggressive: significantly lower minimum improvement threshold
        
        # Compute class prototypes and subclass prototypes
        class_prototypes = compute_class_prototypes(all_image_features_tensor, all_labels_tensor, all_logits, num_classes)
        subclass_prototypes, cluster_info = compute_subclass_prototypes(
            all_image_features_tensor, all_labels_tensor, num_classes, num_clusters=3, min_samples=5
        )
        
        # Display subclass prototype clustering information
        if cluster_info:
            print("\nSubclass prototype clustering information:")
            for c, info in cluster_info.items():
                if 'counts' in info:
                    print(f"Class {c}: {info['n_clusters']} clusters, sample distribution: {info['counts']}")
        
        # Compute cosine similarity between samples and class prototypes and subclass prototypes
        image_features_norm = F.normalize(all_image_features_tensor, p=2, dim=1)
        prototype_sims = torch.matmul(image_features_norm, class_prototypes.t())
        
        # Compute similarity between samples and subclass prototypes (compute similarity for each sample with each class's subclass prototypes)
        subproto_sims = []
        batch_size = 1000  # Compute in batches to avoid memory overflow
        for i in range(0, len(image_features_norm), batch_size):
            end_idx = min(i + batch_size, len(image_features_norm))
            batch_features = image_features_norm[i:end_idx]
            
            # Compute similarity between batch samples and all subclass prototypes
            batch_sims = []
            for c in range(num_classes):
                # Compute similarity with all subclass prototypes of the current class
                c_subprotos = subclass_prototypes[c]  # Shape: [num_clusters, feat_dim]
                # Compute similarity Shape: [batch_size, num_clusters]
                c_sims = torch.matmul(batch_features, c_subprotos.t())
                # Take the maximum similarity for each sample with the class's subclass prototypes
                c_max_sims, _ = torch.max(c_sims, dim=1)  # Shape: [batch_size]
                batch_sims.append(c_max_sims.unsqueeze(1))
            
            # Concatenate similarities for all classes
            batch_subproto_sims = torch.cat(batch_sims, dim=1)  # Shape: [batch_size, num_classes]
            subproto_sims.append(batch_subproto_sims)
        
        # Concatenate subclass similarities for all batches
        subproto_sims = torch.cat(subproto_sims, dim=0)  # Shape: [total_samples, num_classes]
        
        # Initialize statistics
        total_samples = len(all_labels_tensor)
        rescored_samples = 0
        initial_correct = 0
        rescored_correct = 0
        improved_by_rescoring = 0
        worsened_by_rescoring = 0
        
        # Get original prediction results and accuracy
        _, initial_preds = torch.max(all_logits, dim=1)
        initial_correct = (initial_preds == all_labels_tensor).sum().item()
        
        # Initialize rescored predictions
        rescored_preds = initial_preds.clone()
        
        # Custom thresholds and weights for different confused class pairs
        custom_thresholds = {}
        custom_alphas = {}
        custom_improvements = {}
        
        # Special handling for highly confused class pairs
        for i, j, rate in highly_confused_pairs:
            # For severely confused classes, use lower thresholds and higher embedding weights
            custom_thresholds[(i, j)] = uncertainty_threshold * 0.8
            custom_alphas[(i, j)] = max(0.0, alpha - 0.2)  # Further reduce original prediction weight
            custom_improvements[(i, j)] = min_improvement * 0.5  # Lower minimum improvement threshold
        
        # Record improvement statistics for each class
        class_improvements = {c: {'improved': 0, 'worsened': 0, 'total': 0} for c in range(num_classes)}
        
        # Reinforcement learning style class-specific alpha value adjustment
        class_alphas = {c: alpha for c in range(num_classes)}
        
        # Apply three rescoring strategies and vote for each sample
        strategy_votes = torch.zeros((total_samples, num_classes), device=device)
        
        # Strategy 1: Rescoring based on standard prototype similarity
        for i in range(total_samples):
            true_label = all_labels_tensor[i].item()
            pred_label = initial_preds[i].item()
            
            # Record total samples for each class
            if true_label < num_classes:
                class_improvements[true_label]['total'] += 1
            
            # Determine threshold and parameters for the current sample
            curr_threshold = uncertainty_threshold
            curr_alpha = class_alphas.get(true_label, alpha)
            curr_min_improvement = min_improvement
            
            # Check if it is a highly confused class pair
            if pred_label < num_classes and true_label < num_classes:
                pair = (min(true_label, pred_label), max(true_label, pred_label))
                if pair in custom_thresholds:
                    curr_threshold = custom_thresholds[pair]
                    curr_alpha = custom_alphas[pair]
                    curr_min_improvement = custom_improvements[pair]
            
            # If uncertainty exceeds threshold, consider rescoring
            if uncertainty_scores[i] > curr_threshold:
                # Find top-k possible classes
                topk_values, topk_indices = torch.topk(all_logits[i], min(topk_for_rescoring, num_classes))
                
                # Record prediction and similarity before rescoring
                old_pred = initial_preds[i].item()
                
                # Find the class with the highest weighted combined score
                best_score = -1.0
                best_class = -1
                
                for idx in topk_indices:
                    class_idx = idx.item()
                    
                    # Get original prediction probability and embedding similarity
                    orig_prob = probs[i, class_idx]
                    embed_sim = prototype_sims[i, class_idx]
                    
                    # Weighted combined score: alpha*original probability + (1-alpha)*embedding similarity
                    combined_score = curr_alpha * orig_prob + (1 - curr_alpha) * embed_sim
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_class = class_idx
                
                # Record votes for standard strategy
                if best_class >= 0:
                    strategy_votes[i, best_class] += 1.0
        
        # Strategy 2: Rescoring based on subclass prototype similarity
        for i in range(total_samples):
            if uncertainty_scores[i] > uncertainty_threshold:
                # Find top-k possible classes
                topk_values, topk_indices = torch.topk(all_logits[i], min(topk_for_rescoring, num_classes))
                
                # Find the most similar subclass prototype class
                best_score = -1.0
                best_class = -1
                
                for idx in topk_indices:
                    class_idx = idx.item()
                    
                    # Get original prediction probability and subclass prototype similarity
                    orig_prob = probs[i, class_idx]
                    subproto_sim = subproto_sims[i, class_idx]
                    
                    # Weighted combined score, subclass prototype weight higher
                    combined_score = 0.05 * orig_prob + 0.95 * subproto_sim
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_class = class_idx
                
                # Record votes for subclass prototype strategy
                if best_class >= 0:
                    strategy_votes[i, best_class] += 1.5  # Subclass prototype strategy weight higher
        
        # Strategy 3: Special handling for confused class pairs
        confusing_class_map = {}
        for (i, j), info in confusion_pairs.items():
            # If i is more likely to be misclassified as j, map i to j
            if info['i_to_j'] > info['j_to_i'] * 1.5:
                confusing_class_map[i] = j
            # If j is more likely to be misclassified as i, map j to i
            elif info['j_to_i'] > info['i_to_j'] * 1.5:
                confusing_class_map[j] = i
        
        for i in range(total_samples):
            pred_label = initial_preds[i].item()
            
            # Check if it is a highly confused class
            if pred_label in confusing_class_map:
                confused_with = confusing_class_map[pred_label]
                
                # Compare similarity with confused class's subclass prototype
                if subproto_sims[i, confused_with] > subproto_sims[i, pred_label] * 1.1:
                    # If similarity with confused class's subclass prototype is significantly higher, vote for confused class
                    strategy_votes[i, confused_with] += 2.0  # Targeted handling weight highest
        
        # Final decision: Combine votes from three strategies
        for i in range(total_samples):
            # Rescore only if there are enough votes
            if torch.max(strategy_votes[i]) >= 1.0:
                old_pred = initial_preds[i].item()
                new_pred = torch.argmax(strategy_votes[i]).item()
                
                if new_pred != old_pred:
                    rescored_preds[i] = new_pred
                    rescored_samples += 1
                    
                    # Statistics of rescoring effect
                    true_label = all_labels_tensor[i].item()
                    if old_pred == true_label and new_pred != true_label:
                        worsened_by_rescoring += 1
                        if true_label < num_classes:
                            class_improvements[true_label]['worsened'] += 1
                    elif old_pred != true_label and new_pred == true_label:
                        improved_by_rescoring += 1
                        if true_label < num_classes:
                            class_improvements[true_label]['improved'] += 1
        
        # Compute accuracy after rescoring
        rescored_correct = (rescored_preds == all_labels_tensor).sum().item()
        
        # Compute top5 accuracy after rescoring
        rescored_top5_correct = 0
        for i in range(total_samples):
            # Get top5 predictions from original logits
            _, top5_indices = torch.topk(all_logits[i], min(5, num_classes))
            if all_labels_tensor[i] in top5_indices:
                # If true label is in top5, check if rescoring changed this correct prediction
                if rescored_preds[i] != all_labels_tensor[i] and initial_preds[i] == all_labels_tensor[i]:
                    # Rescoring made top1 prediction wrong but retained top5 correctness
                    rescored_top5_correct += 1
                else:
                    # Rescoring did not change this correct top5 prediction
                    rescored_top5_correct += 1
            else:
                # Original top5 predictions do not contain true label, check if rescoring corrected the error
                if rescored_preds[i] == all_labels_tensor[i]:
                    # Rescoring corrected the prediction (at least entered top1)
                    rescored_top5_correct += 1
        
        # Compute original top5 accuracy
        initial_top5_correct = 0
        for i in range(total_samples):
            _, top5_indices = torch.topk(all_logits[i], min(5, num_classes))
            if all_labels_tensor[i] in top5_indices:
                initial_top5_correct += 1
        
        # Print rescoring results
        print("\n===== Original model vs rescoring effect comparison =====")
        print(f"Uncertainty threshold: {uncertainty_threshold:.4f}, Original prediction weight: {alpha:.2f}, Minimum improvement threshold: {min_improvement:.4f}")
        print(f"Total samples: {total_samples}")
        print(f"Rescored samples: {rescored_samples} ({rescored_samples/total_samples*100:.2f}%)")
        print(f"Original accuracy: {initial_correct/total_samples*100:.3f}%")
        print(f"Accuracy after rescoring: {rescored_correct/total_samples*100:.3f}%")
        print(f"Accuracy change: {(rescored_correct-initial_correct)/total_samples*100:+.3f}%")
        print(f"Original Top5 accuracy: {initial_top5_correct/total_samples*100:.3f}%")
        print(f"Top5 accuracy after rescoring: {rescored_top5_correct/total_samples*100:.3f}%")
        print(f"Top5 accuracy change: {(rescored_top5_correct-initial_top5_correct)/total_samples*100:+.3f}%")
        print(f"Improved samples: {improved_by_rescoring}")
        print(f"Worsened samples: {worsened_by_rescoring}")
        
        # Print rescoring effect for each class
        print("\nRescoring effect for each class:")
        for c in range(num_classes):
            if class_improvements[c]['total'] > 0:
                improved = class_improvements[c]['improved']
                worsened = class_improvements[c]['worsened']
                total = class_improvements[c]['total']
                net_change = improved - worsened
                print(f"Class {c}: Total samples={total}, Improved={improved}, Worsened={worsened}, Net change={net_change} ({net_change/total*100:+.2f}%)")
        
        # Add rescoring results to metrics
        metrics['initial_acc'] = initial_correct/total_samples*100
        metrics['rescored_acc'] = rescored_correct/total_samples*100
        metrics['initial_top5_acc'] = initial_top5_correct/total_samples*100
        metrics['rescored_top5_acc'] = rescored_top5_correct/total_samples*100
        metrics['rescored_samples_pct'] = rescored_samples/total_samples*100
        metrics['improved_by_rescoring'] = improved_by_rescoring
        metrics['worsened_by_rescoring'] = worsened_by_rescoring
        metrics['accuracy_change'] = (rescored_correct-initial_correct)/total_samples*100
        metrics['top5_accuracy_change'] = (rescored_top5_correct-initial_top5_correct)/total_samples*100
        
        # Analyze and suggest improvements based on rescoring effect
        if rescored_correct > initial_correct:
            accuracy_improvement = (rescored_correct-initial_correct)/total_samples*100
            print(f"\nSuccess! Rescoring improved accuracy by {accuracy_improvement:.3f}%")
            print(f"Conversion rate: {improved_by_rescoring}/{rescored_samples} = {improved_by_rescoring/max(rescored_samples, 1)*100:.2f}%")
            
            # Add enhanced top1 and top5 accuracy information
            print(f"\n===== Accuracy comparison before and after rescoring =====")
            print(f"Original Top-1 accuracy: {initial_correct/total_samples*100:.3f}%")
            print(f"Enhanced Top-1 accuracy: {rescored_correct/total_samples*100:.3f}% ({(rescored_correct-initial_correct)/total_samples*100:+.3f}%)")
            print(f"Original Top-5 accuracy: {initial_top5_correct/total_samples*100:.3f}%")
            print(f"Enhanced Top-5 accuracy: {rescored_top5_correct/total_samples*100:.3f}% ({(rescored_top5_correct-initial_top5_correct)/total_samples*100:+.3f}%)")
            
            # If improvement is less than 5%, provide further optimization suggestions
            if accuracy_improvement < 5.0:
                print("\nTo further improve accuracy, consider:")
                print("1. Lowering uncertainty threshold to increase rescored sample proportion")
                print("2. Reducing alpha value to further increase embedding similarity weight")
                print("3. Increasing subclass prototype clustering count to better capture intra-class variation")
                print("4. Training or stronger data augmentation for poorly performing classes")
        else:
            print(f"\nWarning: Rescoring reduced accuracy by {(initial_correct-rescored_correct)/total_samples*100:.3f}%")
            print("Consider adjusting parameters or optimizing feature representation.")
    
    # Compute accuracy based on prompt features - only in contrastive learning mode
    if args.main_loss_type in ['infonce', 'cosine_sim', 'ce_infonce'] and len(all_image_features) > 0 and all_prompt_features is not None:
        # Features for subsequent computation
        if not isinstance(all_image_features, torch.Tensor):
            all_image_features = torch.cat(all_image_features, dim=0)
        if not isinstance(all_labels, torch.Tensor):
            all_labels = torch.cat(all_labels, dim=0)
        
        # Compute global Prompt Top-1 accuracy
        if all_prompt_features is not None:
            # Compute similarity based on prompt_features dimension
            all_image_features_norm = F.normalize(all_image_features, p=2, dim=1)
            
            if all_prompt_features.dim() == 3:  # [C, P, D]
                # Compute best prompt similarity for each class
                num_samples = all_image_features_norm.size(0)
                prompt_logits = torch.zeros(num_samples, num_classes, device=all_image_features.device)
                
                # Perform similarity computation on CPU
                all_image_features_norm = all_image_features_norm.to('cpu')
                all_prompt_features = all_prompt_features.to('cpu')
                
                for c in range(num_classes):
                    class_prompts = all_prompt_features[c]  # [P, D]
                    class_prompts_norm = F.normalize(class_prompts, p=2, dim=1)
                    
                    # Compute similarity between each image and all prompts of the class
                    class_sims = torch.matmul(all_image_features_norm, class_prompts_norm.t())
                    # Take maximum similarity
                    max_sims, _ = class_sims.max(dim=1)
                    prompt_logits[:, c] = max_sims
                
                # Compute intra-class and inter-class similarity
                for i in range(num_samples):
                    label = all_labels[i].item()
                    if label < num_classes:
                        # Update intra-class similarity
                        intra_class_sims[label] += prompt_logits[i, label].item()
                        intra_class_counts[label] += 1
                        
                        # Update inter-class maximum similarity
                        for j in range(num_classes):
                            if j != label:
                                sim = prompt_logits[i, j].item()
                                if sim > inter_class_max_sims[label, j]:
                                    inter_class_max_sims[label, j] = sim
            
            else:  # Single prompt case [C, D]
                prompt_features_norm = F.normalize(all_prompt_features, p=2, dim=1)
                # Compute similarity matrix [B, C]
                prompt_logits = torch.matmul(all_image_features_norm, prompt_features_norm.t())
                
                # Compute intra-class and inter-class similarity
                for i in range(all_image_features_norm.size(0)):
                    label = all_labels[i].item()
                    if label < num_classes:
                        # Update intra-class similarity
                        intra_class_sims[label] += prompt_logits[i, label].item()
                        intra_class_counts[label] += 1
                        
                        # Update inter-class maximum similarity
                        for j in range(num_classes):
                            if j != label:
                                sim = prompt_logits[i, j].item()
                                if sim > inter_class_max_sims[label, j]:
                                    inter_class_max_sims[label, j] = sim
            
            # Compute Prompt Top-1 and Top-5 Accuracy
            _, prompt_top1_indices = prompt_logits.topk(1, dim=1)
            prompt_top1_correct = (prompt_top1_indices.squeeze() == all_labels).sum().item()
            prompt_top1_total = all_labels.size(0)
            prompt_top1_acc = 100.0 * prompt_top1_correct / prompt_top1_total
            
            # Compute Prompt Top-5 Accuracy (if number of classes is sufficient)
            if num_classes >= 5:
                _, prompt_top5_indices = prompt_logits.topk(5, dim=1)
                prompt_top5_correct = 0
                for i in range(all_labels.size(0)):
                    if all_labels[i] in prompt_top5_indices[i]:
                        prompt_top5_correct += 1
                prompt_top5_acc = 100.0 * prompt_top5_correct / prompt_top1_total
            else:
                prompt_top5_acc = prompt_top1_acc  # If less than 5 classes, same as Top-1
            
            # Add to metrics dictionary
            metrics['test_prompt_top1_acc'] = prompt_top1_acc
            metrics['test_prompt_top5_acc'] = prompt_top5_acc
            
            # Compute average intra-class similarity
            mean_intra_class_sim = 0.0
            valid_class_count = 0
            for i in range(num_classes):
                if intra_class_counts[i] > 0:
                    mean_intra_class_sim += intra_class_sims[i] / intra_class_counts[i]
                    valid_class_count += 1
            
            if valid_class_count > 0:
                mean_intra_class_sim /= valid_class_count
                metrics['test_mean_intra_class_sim'] = mean_intra_class_sim
            
            # Compute average maximum inter-class similarity
            mean_max_inter_class_sim = 0.0
            valid_pairs = 0
            for i in range(num_classes):
                for j in range(num_classes):
                    if i != j and inter_class_max_sims[i, j] > -1.0:
                        mean_max_inter_class_sim += inter_class_max_sims[i, j]
                        valid_pairs += 1
            
            if valid_pairs > 0:
                mean_max_inter_class_sim /= valid_pairs
                metrics['test_mean_max_inter_class_sim'] = mean_max_inter_class_sim
                
                # Compute intra-inter ratio
                if mean_max_inter_class_sim != 0:
                    intra_inter_ratio = mean_intra_class_sim / mean_max_inter_class_sim
                    metrics['test_intra_inter_ratio'] = intra_inter_ratio
            
            # Print overall model vs Prompt-related metrics comparison
            print("\n===== Model accuracy vs Prompt accuracy comparison =====")
            print(f"Model Top-1 accuracy: {metrics['acc1']:.3f}%")
            print(f"Model Top-5 accuracy: {metrics.get('acc5', 0):.3f}%")
            print(f"Prompt Top-1 accuracy: {prompt_top1_acc:.3f}% ({prompt_top1_correct}/{prompt_top1_total})")
            print(f"Prompt Top-5 accuracy: {prompt_top5_acc:.3f}%")
            print("===== Similarity statistics =====")
            if 'test_mean_intra_class_sim' in metrics:
                print(f"Mean intra-class similarity: {mean_intra_class_sim:.4f}")
            if 'test_mean_max_inter_class_sim' in metrics:
                print(f"Mean maximum inter-class similarity: {mean_max_inter_class_sim:.4f}")
            if 'test_intra_inter_ratio' in metrics:
                print(f"Intra-inter similarity ratio: {intra_inter_ratio:.4f}")
    
    # Ensure all metrics are JSON serializable
    for key in metrics:
        if isinstance(metrics[key], torch.Tensor):
            metrics[key] = metrics[key].cpu().numpy().tolist()
    
    return metrics
