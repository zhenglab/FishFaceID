#!/bin/bash
# ！替换绝对路径
ANACONDA_HOME=/mnt/1Tssd1/video-framework/anaconda310-mamba

export PATH=${ANACONDA_HOME}/bin:$PATH

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}
GPUS=1

SPLIT=$1
BATCH_SIZE=$2
EPOCH=$3
LR=$4
GPU_ID=$5
MIN_LR=$6
WARMUP_LR=$7
SUBSET=$8
INPUT_SIZE=$9
PATCH_SIZE=${10}
STRIDE=${11}
EMBED_DIM=${12}
DATASET=${13}

# @suen 修改工程路径

ROOT_PATH=/mnt/8TDisk1/zhenglab/sunnaizhe/OUC-MOI-ID/vim

# @suen 修改数据集路径

# 1013时
DATA_PATH=/mnt/8TDisk1/zhenglab/sunnaizhe/dataset/OUC-MOI-ID/data
DATA_SET=${DATASET}
MODEL_NAME=pretrain_vim_tiny_input${INPUT_SIZE}_patch${PATCH_SIZE}_stride${STRIDE}_embed${EMBED_DIM}_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

RESULT_PATH=${ROOT_PATH}/output/FinalExpV2/3-TryMamba/PRETRAIN_${DATA_SET}_${SPLIT}_${EPOCH}_${LR}_${MIN_LR}_${WARMUP_LR}_${MODEL_NAME}



# 评估测试
CUDA_VISIBLE_DEVICES=${GPU_ID} python main_snz_exp3_embed_debug.py \
--eval --eval_save \
--test-five-crop \
--resume ${RESULT_PATH}/best_checkpoint.pth \
--subset ${SUBSET}  --species ${DATA_SET} \
--model ${MODEL_NAME} \
--input-size ${INPUT_SIZE} --batch-size ${BATCH_SIZE} --num_workers 1 \
--data-path ${DATA_PATH} \
--data-set ${DATA_SET} \
--output_dir ${RESULT_PATH} \
--no_amp \