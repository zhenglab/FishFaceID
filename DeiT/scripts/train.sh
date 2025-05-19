# 来自上一组实验需修正的：
# main_snz.py
# ？- 固定一个scale
# ？- test是否加five参数


#!/bin/bash
# ！替换绝对路径
ANACONDA_HOME=/mnt/1Tssd1/video-framework/anaconda310-mamba

export PATH=${ANACONDA_HOME}/bin:$PATH
cd /mnt/8TDisk1/zhenglab/sunnaizhe/OUC-MOI-ID/deit-98d
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29500}
GPUS=1

SPLIT=$1
BATCH_SIZE=16
EPOCH=100
LR=1e-3
GPU_ID=$2
MIN_LR=1e-4
WARMUP_LR=1e-5
SUBSET=$3
INPUT_SIZE=224
MODEL_SIZE=$4

# @suen 修改工程路径

ROOT_PATH=/mnt/8TDisk1/zhenglab/sunnaizhe/OUC-MOI-ID/deit-98d

# @suen 修改数据集路径

# 1013时
DATA_PATH=/mnt/8TDisk1/zhenglab/sunnaizhe/dataset/OUC-MOI-ID/data/sea_cucumber
DATA_SET=SeaCum
MODEL_NAME=deit_${MODEL_SIZE}_patch16_224

RESULT_PATH=${ROOT_PATH}/output/final/4-All/PRETRAIN_${DATA_SET}_${SPLIT}_${EPOCH}_${LR}_${MIN_LR}_${WARMUP_LR}_${MODEL_NAME}

# ImageNet finetune
CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
--subset ${SUBSET}  --species ${DATA_SET} \
--model ${MODEL_NAME} \
--input-size ${INPUT_SIZE} --batch-size ${BATCH_SIZE} --num_workers 1 --epochs ${EPOCH} \
--lr ${LR} --min-lr ${MIN_LR} --warmup-lr ${WARMUP_LR} \
--data-path ${DATA_PATH} \
--data-set ${DATA_SET}  --split ${SPLIT} \
--output_dir ${RESULT_PATH} \
--finetune /mnt/8TDisk1/zhenglab/sunnaizhe/OUC-MOI-ID/deit-98d/output/pretrain/deit_${MODEL_SIZE}_patch16_224.pth \

# 评估测试
# CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
# --eval \
# --resume ${RESULT_PATH}/best_checkpoint.pth \
# --model ${MODEL_NAME} \
# --input-size ${INPUT_SIZE} --batch-size ${BATCH_SIZE} --num_workers 1 \
# --data-path ${DATA_PATH} \
# --data-set ${DATA_SET} \
# --output_dir ${RESULT_PATH} \
# --no_amp \