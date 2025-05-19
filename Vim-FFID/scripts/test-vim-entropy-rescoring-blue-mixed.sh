#!/bin/bash

# Change directory to the project root
PROJECT_ROOT="/home/nicozz/Classification/Vim"
cd ${PROJECT_ROOT};

DATA_PATH="/home/nicozz/Data/BlueGrouper/mixed" # Dataset path
# Update output directory name based on sampler and loss function combination
OUTPUT_DIR="./output_use/result_blue_mixed"
NPROC_PER_NODE=1 # Number of GPUs to use
MASTER_PORT=29525 # Unique port for distributed training

# ClassAwareBatchSampler parameters
SAMPLER_P=8       # P: Number of classes per batch
SAMPLER_K=8      # K: Number of samples per class
BATCH_SIZE=$((SAMPLER_P * SAMPLER_K))  # Calculate batch size as P×K

# Multiple prompt parameters
PROMPTS_PER_CLASS=5  # Number of prompts per class

INTERACTION_LAYERS=2  # Number of interaction layers
INTERACTION_STRENGTH=0.3  # Interaction strength

# Advanced Smart Embedding Distance Rescoring parameters - optimization settings
# Set to 0 to use adaptive threshold, automatically adjusted based on data distribution
ENTROPY_THRESHOLD=0
# Increase the number of considered classes to expand rescoring scope
TOPK_FOR_RESCORING=10
# Use environment variable to set rescoring weight factor - more aggressive settings
# Lower alpha value means relying more on embedding similarity rather than original prediction probability
export RESCORING_ALPHA=0.1

echo "Starting ViM-CLIP (BASE) evaluation with Multiple Prompts (${PROMPTS_PER_CLASS}/class) and Advanced Smart Embedding Rescoring"
echo "Using advanced rescoring strategy: adaptive threshold, multi-strategy voting, subprototype clustering, alpha=${RESCORING_ALPHA}"
export LD_PRELOAD=/home/nicozz/anaconda3/envs/mamba/lib/libstdc++.so.6

# Define arguments in a bash array
args=(
    # Model configuration
    --model vim_base_patch16_224_clip_prompts
    --prompts-per-class ${PROMPTS_PER_CLASS}  # Number of prompts per class
    # Base model's backbone parameters are more extensive, try not using intermediate fusion
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
    
    # Enable advanced smart embedding distance rescoring
    --use-entropy-rescoring
    --entropy-threshold ${ENTROPY_THRESHOLD}
    --topk-for-rescoring ${TOPK_FOR_RESCORING}
    
    --data-path "${DATA_PATH}"
    --data-set custom
    --output_dir "${OUTPUT_DIR}"
    --eval
    --resume output_use/bluegrouper_mixed_12_infonce+arcfacefocal+center/checkpoint.pth
    # --resume output_clip_loss_multiprompt_new/vim_base_clip_multiprompts_infonce+arcfacefocal+center+triplet_fish_overhead_class_aware_interaction_inter12/best_checkpoint.pth




    --lr 5e-5
    --min-lr 5e-6
    --warmup-lr 5e-7
    --unscale-lr
    --warmup-epochs 20
    --epochs 300
    --weight-decay 0.05
    --drop-path 0.1
    --clip-grad 1.0
    --model-ema
    --model-ema-decay 0.9999
    --num_workers 4
    --pin-mem
    --no_amp
    --finetune /home/nicozz/Classification/Vim/vim/pretrain/vim_b_midclstok_81p9acc.pth
    
    # Sampler configuration - added ClassAwareBatchSampler
    --batch-sampler class-aware
    --sampler-p ${SAMPLER_P}
    --sampler-k ${SAMPLER_K}
    --use-softmax-balance  # Enable softmax balance, dynamically adjust class weights based on historical loss
    --softmax-temperature 0.1
    
    # Loss function configuration - use the same loss combination as training
    --main-loss-type infonce   # Set main loss function to InfoNCE
    --aux-losses arcface center   # Set auxiliary loss functions
    --aux-weights 1.5 1.0    # Set weights for each auxiliary loss
    --triplet-mining semi-hard  # Set mining strategy for triplet loss
    --infonce-temp 0.05   # Set InfoNCE temperature parameter
    --arcface-s 64.0   # Set ArcFace scaling factor
    --arcface-m 0.5   # Set ArcFace margin angle
    --center-alpha 0.1   # Set learning rate for center loss
    --arcface-use-focal        # Enable Focal Loss in ArcFace
    --arcface-focal-gamma 2.0   # Set Focal Loss gamma (adjustable)
    --arcface-focal-alpha 0.25  # Set Focal Loss alpha (adjustable)

    # Learning rate configuration - lower backbone learning rate for base model
    --prompt-lr-mult 10
    --backbone-lr-mult 1  # Lower backbone learning rate compared to small model
)

# Execute the test script
CUDA_VISIBLE_DEVICES=0 python vim_clip_use/main_clip.py "${args[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Testing completed successfully. Output saved to: ${OUTPUT_DIR}"
    
    # Extract key results
    RESULTS_FILE="${OUTPUT_DIR}/rescoring_results.txt"
    echo "Advanced Smart Embedding Distance Rescoring Results (Multi-strategy Voting + Subprototype Clustering)" > ${RESULTS_FILE}
    echo "===============================" >> ${RESULTS_FILE}
    
    # Extract original accuracy, rescored accuracy and Top5 accuracy
    grep "Original accuracy:" ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    grep "Rescored accuracy:" ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    grep "Accuracy change:" ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    grep "Original Top5 accuracy:" ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    grep "Rescored Top5 accuracy:" ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    grep "Top5 accuracy change:" ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    grep "Improved samples:" ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    grep "Worsened samples:" ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    
    # Extract accuracy comparison before and after rescoring
    echo "" >> ${RESULTS_FILE}
    echo "Accuracy comparison before and after rescoring:" >> ${RESULTS_FILE}
    grep "===== Accuracy comparison before and after rescoring =====" -A 4 ${OUTPUT_DIR}/log.txt >> ${RESULTS_FILE} 2>/dev/null
    
    # Extract class-specific improvement information
    echo "" >> ${RESULTS_FILE}
    echo "Class-specific improvements:" >> ${RESULTS_FILE}
    grep "Class [0-9]:" ${OUTPUT_DIR}/log.txt | grep "total samples=" >> ${RESULTS_FILE} 2>/dev/null
    
    echo "Key results have been saved to: ${RESULTS_FILE}"
    
    # Clear environment variables
    unset RESCORING_ALPHA
else
    echo "Testing failed with exit code: ${EXIT_CODE}"
    
    # Clear environment variables
    unset RESCORING_ALPHA
fi