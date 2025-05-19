#!/bin/bash

# Change directory to the project root
PROJECT_ROOT="/home/nicozz/Classification/Vim"
cd ${PROJECT_ROOT};

DATA_PATH="/home/nicozz/Data/BlueGrouper/mixed" # Dataset path
# Update output directory name based on sampler and loss function combination
OUTPUT_DIR="./output_use/bluegrouper_mixed_12_infonce+arcface+center"
NPROC_PER_NODE=1 # Number of GPUs to use
MASTER_PORT=29525 # Unique port for distributed training

# ClassAwareBatchSampler parameters
SAMPLER_P=8       # P: Number of classes per batch
SAMPLER_K=8      # K: Number of samples per class
BATCH_SIZE=$((SAMPLER_P * SAMPLER_K))  # Calculate batch size as P×K

# Multiple prompt parameters
PROMPTS_PER_CLASS=5  # Number of prompts per class

# Interaction parameters
INTERACTION_LAYERS=2  # Number of interaction layers
INTERACTION_STRENGTH=0.3  # Interaction strength

echo "Starting ViM-CLIP (BASE) training with Multiple Prompts (${PROMPTS_PER_CLASS}/class), Backbone-Prompt Interaction, and Class-Aware Sampling (P=${SAMPLER_P}, K=${SAMPLER_K}, BatchSize=${BATCH_SIZE})"
export LD_PRELOAD=/home/nicozz/anaconda3/envs/mamba/lib/libstdc++.so.6

# Define arguments in a bash array
args=(
    # Model configuration
    --model vim_base_patch16_224_clip_prompts
    --prompts-per-class ${PROMPTS_PER_CLASS}  # Number of prompts per class
    # Base model has more backbone parameters, try not using intermediate fusion
    # Add intermediate layer auxiliary loss
    --use-intermediate-fusion
    --intermediate-layer-idx 12
    --use-intermediate-aux-loss
    --intermediate-aux-loss-layers 12
    --intermediate-loss-type contrastive
    --intermediate-loss-weight 0.25
    --intermediate-temp 0.07

    # Enable backbone-prompt interaction
    --use-cross-interaction
    --interaction-layers ${INTERACTION_LAYERS}
    --interaction-strength ${INTERACTION_STRENGTH}

    --data-path "${DATA_PATH}"
    --data-set custom
    --output_dir "${OUTPUT_DIR}"
    
    # Training parameters
    --batch-size ${BATCH_SIZE}
    --lr 5e-5 #1e-4
    --min-lr 5e-6 #1e-5
    --warmup-lr 5e-7 #1e-6
    --unscale-lr
    --warmup-epochs 20
    --epochs 250
    --weight-decay 0.05
    --drop-path 0.1
    --clip-grad 1.0
    --model-ema
    --model-ema-decay 0.9999
    --num_workers 4
    --pin-mem
    --no_amp
    --finetune /home/nicozz/Classification/Vim/vim/pretrain/vim_b_midclstok_81p9acc.pth
    
    # Sampler configuration - Add ClassAwareBatchSampler
    --batch-sampler class-aware
    --sampler-p ${SAMPLER_P}
    --sampler-k ${SAMPLER_K}
    --use-softmax-balance  # Enable softmax balance, dynamically adjust class weights based on historical loss
    --softmax-temperature 0.1
    
    # Use combined loss functions
    # --main-loss-type infonce
    # --aux-losses arcface center
    # --aux-weights 1.5 1
    # --arcface-s 64.0    # Scaling factor
    # --arcface-m 0.5     # Margin angle (radians)
    # --triplet-mining semi-hard  # Can be 'random'(default), 'hard', or 'semi-hard'
    # --infonce-temp 0.05
    # --center-alpha 0.1

    --main-loss-type infonce   # Set main loss function to InfoNCE
    --aux-losses arcface center  # Set auxiliary loss functions
    --aux-weights 1.5 1.0   # Set weights for each auxiliary loss
    --infonce-temp 0.05   # Set InfoNCE temperature parameter
    --arcface-s 64.0   # Set ArcFace scaling factor
    --arcface-m 0.5   # Set ArcFace margin angle
    --center-alpha 0.1   # Set learning rate for center loss
    # --arcface-use-focal        # Enable Focal Loss in ArcFace
    # --arcface-focal-gamma 2.0   # Set Focal Loss gamma (adjustable)
    # --arcface-focal-alpha 0.25  # Set Focal Loss alpha (adjustable)
    --center-alpha 0.1 
    # --triplet-mining semi-hard  # Set mining strategy for triplet loss


    # Learning rate configuration - Lower backbone learning rate for base model
    --prompt-lr-mult 10
    --backbone-lr-mult 1  # Lower backbone learning rate compared to small model
)

# Execute training script
CUDA_VISIBLE_DEVICES=0 python vim_clip_use/main_clip.py "${args[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully. Output saved to: ${OUTPUT_DIR}"
else
    echo "Training failed with exit code: ${EXIT_CODE}"
fi