# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import sys
from pathlib import Path
from contextlib import suppress
import os
import torch.nn as nn
from timm.models.layers import trunc_normal_

# --- Re-add Handle parent directory import --- 
# Add the parent directory (project root) to the Python path
# Necessary when running script directly, esp. with launch tools
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End Handle parent directory import ---

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

# Use absolute imports relative to the project root (Vim/)
# sys.path modification should make these findable
# import vim_clip.engine_clip as engine_clip  # Comment out or delete
# import vim_clip.losses as losses            # Comment out or delete
# import vim_clip.samplers as samplers        # Comment out or delete
# import vim_clip.augment as augment          # Comment out or delete
# import vim_clip.models_mamba as models_mamba  # Comment out or delete
# import vim_clip.models_mamba_clip as models_mamba_clip # Comment out or delete
# import vim_clip.utils as utils              # Comment out or delete
# import vim_clip.datasets as datasets        # Comment out or delete
# import vim_clip.losses as custom_losses     # Comment out or delete

import engine_clip
import losses
import samplers
import augment
import models_mamba
import models_mamba_clip
import utils
import datasets
# Use the same alias for custom_losses if needed elsewhere, or just use 'losses'
custom_losses = losses

# log about
import mlflow

# Additional import to ensure numpy reliability
import numpy

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script for ViM-CLIP models', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='vim_tiny_patch16_224_clip_prompts', type=str, metavar='MODEL',
                        help='Name of model to train (e.g., vim_tiny_patch16_224_clip_prompts)')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # --- Add Softmax entropy-based embedding distance rescoring parameters ---
    parser.add_argument('--use-entropy-rescoring', action='store_true', default=True,
                        help='Enable Softmax entropy-based embedding distance rescoring (default: True)')
    parser.add_argument('--no-entropy-rescoring', action='store_false', dest='use_entropy_rescoring',
                        help='Disable Softmax entropy-based embedding distance rescoring')
    parser.add_argument('--entropy-threshold', type=float, default=0.5,
                        help='Entropy threshold to trigger embedding distance rescoring (default: 0.5)')
    parser.add_argument('--topk-for-rescoring', type=int, default=5,
                        help='Top-k categories to consider for embedding distance rescoring (default: 5)')
    # --- End Softmax entropy-based embedding distance rescoring parameters ---

    # --- Add arguments for model customization ---
    parser.add_argument('--use-intermediate-fusion', action='store_true', default=False,
                        help='Enable intermediate feature fusion in the model')
    # Allow one or more integers for layer indices
    parser.add_argument('--intermediate-layer-idx', type=int, nargs='*', default=None, 
                        help='Layer indices for intermediate fusion (e.g., 12 or 8 16)')
                        
    # --- Intermediate layer auxiliary loss parameters ---
    parser.add_argument('--use-intermediate-aux-loss', action='store_true', default=False,
                        help='Enable intermediate layer auxiliary loss')
    parser.add_argument('--intermediate-aux-loss-layers', type=int, nargs='*', default=None,
                        help='Layer indices for intermediate auxiliary loss (e.g., 12 or 8 16)')
    parser.add_argument('--intermediate-loss-type', type=str, default='contrastive',
                        choices=['contrastive', 'kl', 'mse'],
                        help='Type of intermediate layer loss')
    parser.add_argument('--intermediate-loss-weight', type=float, default=0.5,
                        help='Weight for intermediate layer loss')
    parser.add_argument('--intermediate-temp', type=float, default=0.07,
                        help='Temperature for intermediate contrastive loss')
    # --- End model customization arguments ---

    parser.add_argument('--model-ema', action='store_true', default=True)
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10)')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". ' + \
                             '"(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1, only used if main_loss_type is ce and mixup is off)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true', default=True)
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')

    parser.add_argument('--train-mode', action='store_true', default=True)
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')

    parser.add_argument('--ThreeAugment', action='store_true') #3augment

    parser.add_argument('--src', action='store_true') #simple random crop

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0, # Default 0.0 as usually incompatible with non-CE losses
                        help="mixup alpha, mixup enabled if > 0. (Only effective if main_loss_type='ce')")
    parser.add_argument('--cutmix', type=float, default=0.0, # Default 0.0 as usually incompatible with non-CE losses
                        help="cutmix alpha, cutmix enabled if > 0. (Only effective if main_loss_type='ce')")
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (Only effective if main_loss_type='ce')")
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Cosub params
    parser.add_argument('--cosub', action='store_true', default=False, # Default False
                        help="Enable CoSub regularization (Only effective if main_loss_type='ce')")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10','CIFAR100', 'IMNET', 'INAT', 'INAT19','custom','CUB200'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='class',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # custom arguments
    parser.add_argument('--if_random_cls_token_position', action='store_true')
    parser.add_argument('--if_random_token_rank', action='store_true')
    parser.add_argument('--if_nan2num', action='store_true')
    parser.add_argument('--if_continue_inf', action='store_true')

    # --- Combined Loss function parameters ---
    parser.add_argument('--main-loss-type', type=str, default='infonce',
                        choices=['infonce', 'cosine_sim', 'ce', 'ce_infonce'],
                        help='Main loss type')
    parser.add_argument('--aux-losses', type=str, nargs='*', default=None,
                        help='Auxiliary losses to use (e.g., center arcface triplet)')
    parser.add_argument('--aux-weights', type=float, nargs='*', default=None,
                        help='Weights for auxiliary losses (must match --aux-losses order)')
    parser.add_argument('--infonce-temp', type=float, default=0.07,
                        help='Temperature for InfoNCE loss')
    parser.add_argument('--triplet-margin', type=float, default=1.0,
                        help='Margin for Triplet loss')
    parser.add_argument('--triplet-mining', type=str, default='random',
                        choices=['random', 'hard', 'semi-hard'],
                        help='Mining strategy for Triplet loss')
    parser.add_argument('--center-alpha', type=float, default=0.5,
                        help='Alpha (learning rate) for Center loss updates (often handled externally)')
    parser.add_argument('--arcface-s', type=float, default=64.0,
                        help='Scale factor (s) for ArcFace/CosFace loss')
    parser.add_argument('--arcface-m', type=float, default=0.50,
                        help='Margin (m) for ArcFace loss')
    parser.add_argument('--cosface-m', type=float, default=0.35,
                        help='Margin (m) for CosFace loss')
    parser.add_argument('--contrast-weight', type=float, default=0.5,
                        help='Weight for the contrastive (InfoNCE) part in ce_infonce loss')
    # --- ArcFace Focal Loss Parameters ---
    parser.add_argument('--arcface-use-focal', action='store_true', default=False,
                        help='Use Focal Loss within ArcFace instead of CrossEntropy')
    parser.add_argument('--arcface-focal-gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss in ArcFace')
    parser.add_argument('--arcface-focal-alpha', type=float, default=0.25,
                        help='Alpha parameter for Focal Loss in ArcFace')
    # --- End Combined Loss function parameters ---

    # First stage freezing parameters
    parser.add_argument('--freeze-backbone', action='store_true', default=False, help='Freeze backbone network')
    parser.add_argument('--freeze-epochs', type=int, default=0, help='Number of epochs to freeze backbone')

    # Added batch-sampler argument
    parser.add_argument('--batch-sampler', type=str, default='random',
                        choices=['random', 'balanced', 'class-aware'],
                        help='Batch sampler to use (random, balanced, or class-aware)')
    
    # Add ClassAwareBatchSampler related parameters
    parser.add_argument('--sampler-p', type=int, default=8,
                        help='Number of classes per batch for class-aware sampler')
    parser.add_argument('--sampler-k', type=int, default=8,
                        help='Number of samples per class for class-aware sampler')
    parser.add_argument('--use-softmax-balance', action='store_true', default=False,
                        help='Enable softmax balance for class-aware sampler')
    parser.add_argument('--softmax-temperature', type=float, default=1.0,
                        help='Temperature for softmax-balance class sampling')

    # Backbone-Prompt interaction parameters
    parser.add_argument('--use-cross-interaction', action='store_true', default=False,
                        help='Enable cross-attention interaction between backbone features and prompt features')
    parser.add_argument('--interaction-layers', default=2, type=int,
                        help='Number of layers in the interaction module')
    parser.add_argument('--interaction-strength', default=0.3, type=float,
                        help='Interaction strength coefficient, controlling the influence of interaction features')

    # --- Add arguments for learning rate multipliers ---
    parser.add_argument('--prompts-per-class', type=int, default=1,
                        help='Number of prompts per class')
    parser.add_argument('--prompt-lr-mult', type=float, default=1.0,
                        help='Multiplier for prompt learning rate')
    parser.add_argument('--backbone-lr-mult', type=float, default=1.0,
                        help='Multiplier for backbone learning rate')
    # --- End learning rate multipliers ---

    return parser


def main(args):
    # Import numpy again within the function
    import numpy as np
    
    utils.init_distributed_mode(args)

    print(args)

    # --- Validate Aux Loss Args ---
    if args.aux_losses and args.aux_weights:
        if len(args.aux_losses) != len(args.aux_weights):
            raise ValueError("Number of --aux-losses must match number of --aux-weights.")
    elif args.aux_losses and not args.aux_weights:
        print("Warning: --aux-losses provided without --aux-weights. Using default weight 1.0 for all aux losses.")
    elif not args.aux_losses and args.aux_weights:
        print("Warning: --aux-weights provided without --aux-losses. Ignoring --aux-weights.")
        args.aux_weights = None
    # --- End Validation ---

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # Use the imported datasets module
    dataset_train, args.nb_classes = datasets.build_dataset(is_train=True, args=args)
    dataset_val, _ = datasets.build_dataset(is_train=False, args=args)

    # Debug: Print dataset type and attributes
    print(f"Dataset train type: {type(dataset_train)}")
    try:
        print(f"Dataset train targets length: {len(dataset_train.targets)}")
    except AttributeError:
        print("Dataset train does not have 'targets' attribute.")
    try:
        print(f"Dataset train labels length: {len(dataset_train.get_labels())}") # Assuming get_labels exists
    except AttributeError:
        print("Dataset train does not have 'get_labels' method.")

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = samplers.RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. \n'
                      'This will pad the validation set.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Handle specific sampler configurations
    if args.batch_sampler == 'balanced':
        # Get dataset labels
        if hasattr(dataset_train, 'targets'):
            labels = dataset_train.targets 
        elif hasattr(dataset_train, 'get_labels'):
            labels = dataset_train.get_labels()
        else:
            print("Warning: Dataset does not have targets attribute or get_labels method.")
            print("Falling back to random sampling.")
            args.batch_sampler = 'random'
            
        if args.batch_sampler == 'balanced':
            # Calculate sample counts per class
            class_counts = np.bincount(labels)
            # Calculate weights
            weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            # Assign weights based on sample labels
            sample_weights = weights[labels]
            # Create sampler
            sampler_train = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
    elif args.batch_sampler == 'class-aware' and not args.eval:
        # class-aware sampler is only used for training, not evaluation
        batch_sampler = samplers.ClassAwareBatchSampler(
            dataset=dataset_train,
            num_classes_per_batch=args.sampler_p,
            num_samples_per_class=args.sampler_k,
            use_softmax_balance=args.use_softmax_balance,
            temperature=args.softmax_temperature,
            seed=args.seed
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
        
        if args.ThreeAugment:
            data_loader_train.dataset.transform = augment.new_data_aug_generator(args)
            
        # Create validation data loader
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=True,  # Add shuffle=True to force random order
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        
        # Return early to skip standard DataLoader creation below
        goto_mixup_setup = True
    else:
        # Default to random sampler
        goto_mixup_setup = False

    # Only create standard DataLoader if not using class-aware
    if not goto_mixup_setup:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        if args.ThreeAugment:
            data_loader_train.dataset.transform = augment.new_data_aug_generator(args)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=True,  # Add shuffle=True to force random order
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    # --- Mixup Setup --- Disable Mixup if main loss is not 'ce'
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if args.main_loss_type != 'ce' and mixup_active:
        print(f"Warning: Disabling Mixup/Cutmix as main_loss_type is '{args.main_loss_type}' (not 'ce').")
        mixup_active = False # Override based on main loss type
    elif mixup_active: # Only possible if main_loss_type is 'ce'
        print("Mixup/Cutmix enabled (main_loss_type is 'ce').")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes
        )

    print(f"Creating model: {args.model}")
    
    # --- Build keyword arguments for model creation ---
    model_kwargs = {}
    
    # Handle intermediate fusion and layer indices if provided
    if args.use_intermediate_fusion:
        model_kwargs['use_intermediate_fusion'] = True
        if args.intermediate_layer_idx:
            model_kwargs['intermediate_layer_idx'] = args.intermediate_layer_idx
            print(f"Using intermediate fusion at layers: {args.intermediate_layer_idx}")
        else:
            print("Warning: use_intermediate_fusion=True but no layer indices provided.")
    
    # Add intermediate layer auxiliary loss parameters
    if args.use_intermediate_aux_loss:
        model_kwargs['use_intermediate_aux_loss'] = True
        if args.intermediate_aux_loss_layers:
            model_kwargs['intermediate_aux_loss_layers'] = args.intermediate_aux_loss_layers
            print(f"Using intermediate auxiliary loss at layers: {args.intermediate_aux_loss_layers}")
        else:
            print("Warning: use_intermediate_aux_loss=True but no layer indices provided.")
    
    # Add model output and prompt diversity parameters
    if hasattr(args, 'prompts_per_class'):
        model_kwargs['prompts_per_class'] = args.prompts_per_class
        print(f"Setting prompts_per_class={args.prompts_per_class}")
    
    if hasattr(args, 'prompt_diversity_weight'):
        model_kwargs['prompt_diversity_weight'] = args.prompt_diversity_weight
        print(f"Setting prompt_diversity_weight={args.prompt_diversity_weight}")
    
    # Add parameters for bidirectional Mamba
    if hasattr(args, 'if_bidirectional'):
        model_kwargs['if_bidirectional'] = args.if_bidirectional
        print(f"Setting if_bidirectional={args.if_bidirectional}")
        
    # Add BiMamba type parameter
    if hasattr(args, 'bimamba_type'):
        model_kwargs['bimamba_type'] = args.bimamba_type
        print(f"Setting bimamba_type={args.bimamba_type}")
    
    # Process interaction parameters before model creation
    if args.use_cross_interaction:
        model_kwargs['use_cross_interaction'] = True
        model_kwargs['interaction_layers'] = args.interaction_layers
        model_kwargs['interaction_strength'] = args.interaction_strength
        print(f"Enable backbone-prompt interaction: {args.interaction_layers} layers, strength={args.interaction_strength}")
    
    model = create_model(
        args.model,
        pretrained=False, # Assuming no pretrained weights for these custom models via timm
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None, # Often not used in ViT/Mamba
        img_size=args.input_size,
        **model_kwargs # Pass the collected arguments here
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        # Get state dict of the potentially DDP-wrapped model module
        model_without_ddp_local = model.module if hasattr(model, 'module') else model 
        state_dict = model_without_ddp_local.state_dict()

        # --- Improved head key handling --- 
        # Check if the current model uses class aware prompts
        # This logic now also influences whether CE loss can be used directly
        # If using prompts, model outputs features, so main_loss_type likely needs to be infonce/cosine_sim
        # If NOT using prompts, model outputs logits, so main_loss_type likely needs to be ce
        uses_prompts = getattr(model_without_ddp_local, 'use_class_aware_prompts', False)
        if uses_prompts:
            print("Model uses class-aware prompts. Expecting feature outputs (suitable for infonce/cosine_sim main loss).")
            if args.main_loss_type == 'ce':
                print("Warning: Model uses prompts (feature output) but main_loss_type is 'ce' (expects logits). This might lead to errors unless CombinedLoss handles it.")
            # Remove standard head keys from the checkpoint if they exist.
            print("Removing standard head keys from checkpoint for prompt-based model.")
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint.")
                    del checkpoint_model[k]
        else:
            print("Model does NOT use class-aware prompts. Expecting logit outputs (suitable for ce main loss).")
            if args.main_loss_type != 'ce':
                 print(f"Warning: Model outputs logits, but main_loss_type is '{args.main_loss_type}'. CombinedLoss might expect features.")
            # If not using prompts, current model should have a standard head.
            # Check for shape mismatches or missing keys in the current model.
            for k in ['head.weight', 'head.bias']:#, 'head_dist.weight', 'head_dist.bias']:
                 # Check for head_dist keys only if model actually has them
                 if k in state_dict:
                     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                         print(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
                         del checkpoint_model[k]
                 elif k in checkpoint_model:
                     print(f"Warning: Key {k} found in checkpoint but not in the current standard model. Removing from checkpoint.")
                     del checkpoint_model[k]
                 # Handle head_dist similarly if it exists
                 dist_k = k.replace('head', 'head_dist')
                 if dist_k in state_dict:
                     if dist_k in checkpoint_model and checkpoint_model[dist_k].shape != state_dict[dist_k].shape:
                          print(f"Removing key {dist_k} from pretrained checkpoint due to shape mismatch.")
                          del checkpoint_model[dist_k]
                 elif dist_k in checkpoint_model:
                     print(f"Warning: Key {dist_k} found in checkpoint but not in the current standard model. Removing from checkpoint.")
                     del checkpoint_model[dist_k]
        # --- End improved head key handling ---

        # interpolate position embedding (check compatibility with prompts)
        # If using prompts, absolute pos embed might be disabled. This code might need adjustment.
        if 'pos_embed' in checkpoint_model and hasattr(model_without_ddp_local, 'pos_embed') and model_without_ddp_local.pos_embed is not None:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            # Use patch_embed from the potentially wrapped model
            patch_embed_local = model_without_ddp_local.patch_embed
            num_patches = patch_embed_local.num_patches
            num_extra_tokens = model_without_ddp_local.pos_embed.shape[-2] - num_patches
            # Handle cases where checkpoint might not have extra tokens (e.g., no CLS token)
            ckpt_num_extra_tokens = pos_embed_checkpoint.shape[-2] - (pos_embed_checkpoint.shape[-2] - num_extra_tokens if pos_embed_checkpoint.shape[-2] >= num_extra_tokens else 0)
            if ckpt_num_extra_tokens != num_extra_tokens:
                 print(f"Warning: Mismatch in number of extra tokens ({ckpt_num_extra_tokens} vs {num_extra_tokens}). Pos embed interpolation might be incorrect.")

            orig_size = int((pos_embed_checkpoint.shape[-2] - ckpt_num_extra_tokens) ** 0.5)
            new_size = int(num_patches ** 0.5)
            if orig_size != new_size:
                print(f"Interpolating pos_embed from {orig_size}x{orig_size} to {new_size}x{new_size}")
                # Ensure we only take the expected number of extra tokens from checkpoint
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens] if ckpt_num_extra_tokens >= num_extra_tokens else pos_embed_checkpoint
                pos_tokens = pos_embed_checkpoint[:, ckpt_num_extra_tokens:]
                # Check if pos_tokens is empty
                if pos_tokens.shape[1] > 0:
                     pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                     pos_tokens = torch.nn.functional.interpolate(
                         pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                     pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                     new_pos_embed = torch.cat((extra_tokens[:, :num_extra_tokens], pos_tokens), dim=1) # Slice extra_tokens again just in case
                else:
                     print("Warning: No positional tokens found in checkpoint after extracting extra tokens. Using only extra tokens.")
                     new_pos_embed = extra_tokens[:, :num_extra_tokens]

                checkpoint_model['pos_embed'] = new_pos_embed
            #else: # Don't print this if pos_embed doesn't exist
            #     print("Skipping pos_embed interpolation as sizes match.")
        elif 'pos_embed' in checkpoint_model:
            print("Removing 'pos_embed' from checkpoint as current model does not have it or it's None.")
            del checkpoint_model['pos_embed']
            
        # Add special handling for patch_embed.proj mismatches
        if hasattr(model_without_ddp_local, 'patch_embed') and hasattr(model_without_ddp_local.patch_embed, 'proj'):
            # Check if current model and pretrained model have different patch sizes
            if 'patch_embed.proj.weight' in checkpoint_model and checkpoint_model['patch_embed.proj.weight'].shape != model_without_ddp_local.patch_embed.proj.weight.shape:
                print(f"Detected mismatch in patch_embed.proj.weight shape!")
                print(f"Pretrained model: {checkpoint_model['patch_embed.proj.weight'].shape}")
                print(f"Current model: {model_without_ddp_local.patch_embed.proj.weight.shape}")
                print("Removing patch_embed.proj related parameters from checkpoint, these will be reinitialized")
                
                # Remove all parameters related to patch_embed.proj
                keys_to_remove = [k for k in checkpoint_model.keys() if 'patch_embed.proj' in k]
                for k in keys_to_remove:
                    print(f"Removing: {k}")
                    del checkpoint_model[k]

        # Load the modified checkpoint state dict into the base model (before potential DDP wrapping)
        msg = model_without_ddp_local.load_state_dict(checkpoint_model, strict=False)
        print("Finetuning state dict loading message:", msg)

        # Ensure multi-prompt settings are correctly applied
        if hasattr(model_without_ddp_local, 'use_class_aware_prompts') and model_without_ddp_local.use_class_aware_prompts:
            original_prompts_per_class = model_without_ddp_local.prompts_per_class
            if original_prompts_per_class != args.prompts_per_class:
                print(f"Correcting prompts_per_class parameter: updating from {original_prompts_per_class} to {args.prompts_per_class}")
                # Update model attribute
                model_without_ddp_local.prompts_per_class = args.prompts_per_class
                
                # If prompt_tokens shape does not match expectation, recreate and initialize
                if hasattr(model_without_ddp_local, 'prompt_tokens'):
                    expected_shape = (1, args.nb_classes, args.prompts_per_class, model_without_ddp_local.embed_dim)
                    current_shape = model_without_ddp_local.prompt_tokens.shape
                    
                    if current_shape != expected_shape:
                        print(f"Recreating prompt_tokens parameter: from {current_shape} to {expected_shape}")
                        # Save old prompt values for initialization
                        old_prompts = model_without_ddp_local.prompt_tokens.data
                        
                        # Create new prompt parameter
                        model_without_ddp_local.prompt_tokens = nn.Parameter(
                            torch.zeros(expected_shape, device=device, dtype=model_without_ddp_local.prompt_tokens.dtype)
                        )
                        
                        # Initialize new prompts - copy existing values and randomly initialize the rest
                        with torch.no_grad():
                            # Copy existing values
                            min_classes = min(old_prompts.shape[1], args.nb_classes)
                            min_prompts = min(old_prompts.shape[2], args.prompts_per_class)
                            model_without_ddp_local.prompt_tokens.data[0, :min_classes, :min_prompts] = old_prompts[0, :min_classes, :min_prompts]
                            
                            # Initialize additional prompts
                            if min_prompts < args.prompts_per_class:
                                trunc_normal_(model_without_ddp_local.prompt_tokens.data[0, :, min_prompts:], std=.02)
                                print(f"Randomly initialized {args.prompts_per_class - min_prompts} additional prompts")

        if args.attn_only:
            print("Setting requires_grad=False for non-attention parameters.")
            for name_p,p in model_without_ddp_local.named_parameters():
                if 'attn' not in name_p:
                    p.requires_grad = False
            try:
                if hasattr(model_without_ddp_local, 'head') and model_without_ddp_local.head is not None:
                     model_without_ddp_local.head.weight.requires_grad = True
                     model_without_ddp_local.head.bias.requires_grad = True
                     print("Set requires_grad=True for head parameters.")
                elif hasattr(model_without_ddp_local, 'fc') and model_without_ddp_local.fc is not None: # Check for fc alternative
                     model_without_ddp_local.fc.weight.requires_grad = True
                     model_without_ddp_local.fc.bias.requires_grad = True
                     print("Set requires_grad=True for fc parameters.")
                # Note: Prompt-based models might not have a standard head/fc

            except AttributeError:
                print('Could not find standard head/fc to set requires_grad=True.')

            try:
                if hasattr(model_without_ddp_local, 'pos_embed') and model_without_ddp_local.pos_embed is not None:
                    model_without_ddp_local.pos_embed.requires_grad = True
                    print("Set requires_grad=True for pos_embed.")
            except AttributeError:
                print('no pos_embed found or requires_grad cannot be set.')
            # Try to make final norm/fc layers trainable too
            try:
                for name_p,p in model_without_ddp_local.named_parameters():
                    # Be more specific to avoid enabling all fc layers in blocks
                    if name_p.startswith('norm.') or name_p.startswith('fc_norm.'): # Often final norm before head
                         p.requires_grad = True
                         print(f"Set requires_grad=True for {name_p}.")
                    # Add other potential final layer names if needed
            except Exception as e:
                print(f'Error setting requires_grad for final layers: {e}')

    # --- Move model to device BEFORE DDP wrapping --- 
    model.to(device)

    # Assign model_without_ddp *after* potential finetuning and moving to device
    model_without_ddp = model # No need for module check here as it's before DDP
    
    # --- Get Embed Dim AFTER model creation/loading ---
    try:
         # Attempt to get embed_dim from the model (might be specific to VisionMamba)
         embed_dim = model_without_ddp.embed_dim
         print(f"Model embedding dimension detected: {embed_dim}")
    except AttributeError:
         print("Warning: Could not automatically determine model embedding dimension ('embed_dim'). Check model implementation.")
         # Fallback or raise error - For now, let CombinedLoss potentially fail if needed.
         embed_dim = None # Set to None, CombinedLoss init might handle this or raise error

    if args.distributed:
        print(f"Wrapping model with DDP on GPU {args.gpu}")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True) # find_unused_parameters=True might be needed
        model_without_ddp = model.module # Now get the underlying module after DDP
    # else: # model_without_ddp is already assigned correctly
    #     pass 
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # --- Create EMA model AFTER moving base model to device --- 
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda()
        model_ema = ModelEma(
            model_without_ddp, # Use the unwrapped model for EMA
            decay=args.model_ema_decay, 
            device='cpu' if args.model_ema_force_cpu else '', 
            resume=''
            )

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    # Define parameter groups - insert before create_optimizer
    prompt_params = []
    backbone_params = []

    # Group parameters by name
    for name, param in model_without_ddp.named_parameters():
        if 'prompt_tokens' in name:
            prompt_params.append(param)
            print(f"Added to prompt group: {name}")
        else:
            backbone_params.append(param)
            print(f"Added to backbone group: {name}")

    # Print parameter counts for verification
    print(f"Prompt parameter count: {sum(p.numel() for p in prompt_params)}")
    print(f"Backbone parameter count: {sum(p.numel() for p in backbone_params)}")

    # Create parameter group dictionary
    # Ensure these two parameters exist in args, use default value 1.0 if not
    backbone_lr_mult = getattr(args, 'backbone_lr_mult', 1.0)
    prompt_lr_mult = getattr(args, 'prompt_lr_mult', 1.0)
    
    param_groups = [
        {'params': backbone_params, 'lr': args.lr * backbone_lr_mult},
        {'params': prompt_params, 'lr': args.lr * prompt_lr_mult}
    ]

    # Create optimizer using custom parameter groups
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # support amp
    amp_autocast = suppress
    loss_scaler = None
    # Check for AMP argument correctly (assuming it's added to parser later)
    if getattr(args, 'if_amp', False): # Use getattr for safety
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        print("AMP Enabled")
    else:
        print("AMP Disabled")


    lr_scheduler, _ = create_scheduler(args, optimizer)

    # --- Initialize Loss Function --- 
    if args.main_loss_type == 'ce' and not args.mixup > 0 and not args.cutmix > 0 and not args.cutmix_minmax:
        # Only use smoothing if main loss is CE and mixup/cutmix are off
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        print(f"Using main loss: Label Smoothing Cross Entropy (smoothing={args.smoothing})")
    elif args.mixup > 0. or args.cutmix > 0. or args.cutmix_minmax is not None:
        # SoftTargetCE is needed for mixup/cutmix, but only works if model outputs logits
        if args.main_loss_type not in ['ce', 'ce_infonce']:
             print(f"Warning: Mixup/Cutmix enabled, but main loss '{args.main_loss_type}' might not be compatible. Expected logits output.")
             # Proceeding, assuming downstream loss handles features/logits correctly
             criterion = SoftTargetCrossEntropy()
             print("Using main loss: Soft Target Cross Entropy (for Mixup/Cutmix)")
        else:
             criterion = SoftTargetCrossEntropy()
             print("Using main loss: Soft Target Cross Entropy (for Mixup/Cutmix)")
    elif args.distillation_type != 'none' and not args.bce_loss:
        # If distillation is used (and not BCE), it wraps the base criterion
        # The base criterion is determined inside DistillationLoss if needed
        # For now, we can perhaps set a default CE base criterion here if needed
        # but DistillationLoss might handle its own base based on output type.
        # Let's assume DistillationLoss setup handles the base loss. We set criterion later.
        criterion = None # To be wrapped by DistillationLoss
        print("Distillation enabled, criterion will be handled by DistillationLoss wrapper.")
    elif args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Using main loss: BCEWithLogitsLoss")
    else:
        # Default: Initialize CombinedLoss based on args
        aux_weights_dict = {}
        if args.aux_losses and args.aux_weights:
            if len(args.aux_losses) != len(args.aux_weights):
                raise ValueError("--aux-losses and --aux-weights must have the same number of elements")
            aux_weights_dict = {name: weight for name, weight in zip(args.aux_losses, args.aux_weights)}
            
        # Add intermediate loss to aux_losses if enabled
        effective_aux_losses = list(args.aux_losses) if args.aux_losses else [] # Make a copy
        if args.use_intermediate_aux_loss:
            if 'intermediate' not in effective_aux_losses:
                effective_aux_losses.append('intermediate')
            # Add weight for intermediate loss if not already provided
            if 'intermediate' not in aux_weights_dict:
                aux_weights_dict['intermediate'] = args.intermediate_loss_weight

        # Prepare kwargs for CombinedLoss, including new Focal Loss params
        loss_kwargs = {
            'infonce_temp': args.infonce_temp,
            'triplet_margin': args.triplet_margin,
            'triplet_mining': args.triplet_mining,
            'center_alpha': args.center_alpha,
            'arcface_s': args.arcface_s,
            'arcface_m': args.arcface_m,
            'cosface_s': args.arcface_s, # Use the same scale for CosFace for simplicity
            'cosface_m': args.cosface_m,
            'contrast_weight': args.contrast_weight,
            # Intermediate loss params
            'intermediate_loss_type': args.intermediate_loss_type,
            'intermediate_temp': args.intermediate_temp,
            # ArcFace Focal Loss params
            'arcface_use_focal': args.arcface_use_focal,
            'arcface_focal_gamma': args.arcface_focal_gamma,
            'arcface_focal_alpha': args.arcface_focal_alpha,
        }
        
        criterion = custom_losses.CombinedLoss(
            embed_dim=model.embed_dim,
            num_classes=args.nb_classes,
            main_loss_type=args.main_loss_type,
            aux_losses=effective_aux_losses,
            aux_weights=aux_weights_dict,
            device=args.device,
            **loss_kwargs # Pass all parameters via kwargs
        )
        print(f"Initialized CombinedLoss. Main: {args.main_loss_type}, Aux: {effective_aux_losses}, Weights: {aux_weights_dict}")

    # Setup distillation if needed (wraps the existing criterion)
    if args.distillation_type != 'none':
        assert args.teacher_path,'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        # Handle potential key mismatches (e.g., 'model' vs top level)
        if 'model' in checkpoint:
             teacher_model.load_state_dict(checkpoint['model'])
        else:
             teacher_model.load_state_dict(checkpoint)
        teacher_model.to(device)
        teacher_model.eval()

        print("Wrapping criterion with DistillationLoss.")
        criterion = losses.DistillationLoss(
            criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
        )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # Use model_without_ddp for loading state dict
        load_msg = model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume state dict loading message:", load_msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema and 'model_ema' in checkpoint:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            # Load scaler state if AMP was used and scaler exists
            if getattr(args, 'if_amp', False) and 'scaler' in checkpoint and loss_scaler is not None and checkpoint['scaler'] is not None:
                 print("Loading AMP scaler state.")
                 loss_scaler.load_state_dict(checkpoint['scaler'])
            elif getattr(args, 'if_amp', False):
                 print("Warning: Resuming with AMP enabled, but no scaler state found in checkpoint or scaler is None.")

    if args.eval:
        # Initialize epoch variable to 0, or retrieve from checkpoint
        epoch = 0
        max_accuracy = 0.0
        if args.resume and isinstance(args.resume, str) and os.path.isfile(args.resume):
            try:
                checkpoint = torch.load(args.resume, map_location='cpu')
                if 'epoch' in checkpoint:
                    epoch = checkpoint['epoch']
            except Exception as e:
                print(f"Unable to retrieve epoch from checkpoint: {e}")
        
        # Add test dataset diagnostics
        print("\n----- Test Dataset Diagnostics -----")
        print(f"Test dataset type: {type(dataset_val)}")
        print(f"Test dataset size: {len(dataset_val)}")
        print(f"Total number of classes: {args.nb_classes}")
        
        # Check class distribution
        if hasattr(dataset_val, 'targets'):
            class_counts = np.bincount(dataset_val.targets, minlength=args.nb_classes)
            print(f"Dataset class distribution: {class_counts}")
        
            # Output warning if any class has no samples
            for cls_idx, count in enumerate(class_counts):
                if count == 0:
                    print(f"Warning: Class {cls_idx} has no samples in the test dataset!")
                else:
                    print(f"Class {cls_idx}: {count} samples")
        
        # Check samples in DataLoader
        print("\nChecking first 100 sample classes in DataLoader...")
        class_in_loader = []
        for i, (_, target) in enumerate(data_loader_val):
            class_in_loader.extend(target.tolist())
            if i >= 100 // args.batch_size:  # Only check first 100 samples
                break
        
        loader_class_counts = np.bincount(class_in_loader, minlength=args.nb_classes)
        print(f"Class distribution in DataLoader: {loader_class_counts}")
        
        # Pass args to evaluate
        test_stats = engine_clip.evaluate(data_loader_val, model, device, amp_autocast=amp_autocast, criterion=criterion, args=args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%")

        # New: Save detailed per-class accuracy to file
        if utils.is_main_process() and args.output_dir:
            # Save current epoch results
            per_class_results = {
                'epoch': epoch,
                'per_class_acc': test_stats['per_class_acc'],
                'per_class_total': test_stats['per_class_total'],
                'overall_acc': test_stats['acc1']
            }
            with open(os.path.join(args.output_dir, f'per_class_stats_epoch{epoch}.json'), 'w') as f:
                json.dump(per_class_results, f, indent=2)
            
            # Save best model results separately
            if max_accuracy < test_stats['acc1']:
                with open(os.path.join(args.output_dir, 'best_per_class_stats.json'), 'w') as f:
                    json.dump(per_class_results, f, indent=2)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    #log
    if utils.is_main_process(): # Only log from main process
         try:
              run_name = f'{args.model}_{args.main_loss_type}'
              if args.aux_losses:
                   run_name += f"_aux-{'_'.join(args.aux_losses)}"
              run_name += f'_{time.strftime("%Y%m%d-%H%M%S")}'
              mlflow.start_run(run_name=run_name)
              # log params
              for key, value in vars(args).items():
                  mlflow.log_param(key, str(value)) # Ensure value is string for mlflow
              print("MLflow logging started.")
         except Exception as e:
              print(f"MLflow initialization failed: {e}")

    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate for the current epoch
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        elif hasattr(data_loader_train, 'batch_sampler') and hasattr(data_loader_train.batch_sampler, 'set_epoch'):
            # Support ClassAwareBatchSampler
            data_loader_train.batch_sampler.set_epoch(epoch)
        elif hasattr(data_loader_train, 'sampler') and hasattr(data_loader_train.sampler, 'set_epoch'):
            # Support other custom samplers
            data_loader_train.sampler.set_epoch(epoch)

        # First stage freezing logic
        if args.freeze_backbone or epoch < args.freeze_epochs:
            # Freeze backbone network
            for name, param in model.named_parameters():
                if 'prompt' not in name and 'head' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(f"Freezing backbone network, only training prompts/classification head - Epoch {epoch}")

        train_stats = engine_clip.train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, amp_autocast, args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode during finetuning
            args = args # Pass args here
            # Removed loss_type argument
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            save_state = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if model_ema:
                 save_state['model_ema'] = get_state_dict(model_ema)
            if loss_scaler:
                 save_state['scaler'] = loss_scaler.state_dict()

            for checkpoint_path in checkpoint_paths:
                 utils.save_on_master(save_state, checkpoint_path)
             
        # evaluate
        # Pass args to evaluate
        test_stats = engine_clip.evaluate(data_loader_val, model, device, amp_autocast=amp_autocast, criterion=criterion, args=args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%")
        
        current_acc = test_stats.get("acc1", 0.0) # Use .get for safety
        if max_accuracy < current_acc:
            max_accuracy = current_acc
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                # Reuse save_state from above
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master(save_state, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.3f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        # log metrics to mlflow (only on main process)
        if utils.is_main_process():
             try:
                  mlflow.log_metric("train_loss", train_stats['loss'], step=epoch)
                  mlflow.log_metric("test_loss", test_stats['loss'], step=epoch)
                  mlflow.log_metric("test_acc1", test_stats['acc1'], step=epoch)
                  mlflow.log_metric("test_acc5", test_stats['acc5'], step=epoch)
                  mlflow.log_metric("max_accuracy", max_accuracy, step=epoch)
                  mlflow.log_metric("lr", train_stats['lr'], step=epoch)
             except Exception as e:
                  print(f"MLflow logging failed for epoch {epoch}: {e}")

        # Add helper function to convert Tensor to serializable type
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                serializable_log_stats = convert_to_serializable(log_stats)
                f.write(json.dumps(serializable_log_stats) + "\n")

        # Second stage thawing logic
        if epoch >= args.freeze_epochs:
            # Thaw all parameters
            for param in model.parameters():
                param.requires_grad = True
            # Lower learning rate (optional)
            if epoch == args.freeze_epochs:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.1  # Lower learning rate
            print(f"Thaw all layers, global fine-tuning - Epoch {epoch}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
         try:
              mlflow.end_run()
         except Exception as e:
              print(f"MLflow end_run failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViM-CLIP Combined Loss Training Script', parents=[get_args_parser()])
    # custom args for AMP
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=False) # Default AMP to False unless specified

    args = parser.parse_args()
    # Removed check for infonce_temp as it's handled within CombinedLoss args
    # Check for embed_dim necessity (already handled inside main where criterion is created)
    # if args.aux_losses and any(l in ['center', 'arcface', 'cosface'] for l in args.aux_losses):
    #      print("Note: Aux losses Center/ArcFace/CosFace require the model to have an 'embed_dim' attribute.")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
