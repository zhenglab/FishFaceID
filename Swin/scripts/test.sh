#!/bin/bash
# ！替换绝对路径
ANACONDA_HOME=/mnt/1Tssd1/video-framework/anaconda310/envs/vmamba

export PATH=${ANACONDA_HOME}/bin:$PATH
export LOCAL_RANK=0
export WORLD_SIZE=1

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29501}
GPUS=1

SPLIT=$1
BATCH_SIZE=$2
GPU_ID=$3
SUBSET=$4
MODEL_SIZE=$5
DATA_SET=$6
PORT=$7

cd /mnt/8TDisk1/zhenglab/sunnaizhe/OUC-MOI-ID/SwinTrans-dev


# @suen 修改工程路径

ROOT_PATH=/mnt/8TDisk1/zhenglab/sunnaizhe/OUC-MOI-ID/SwinTrans-dev

# @suen 修改数据集路径

# 1013时
DATA_PATH=/mnt/8TDisk1/zhenglab/sunnaizhe/dataset/OUC-MOI-ID/data
MODEL_NAME=pre_vim_${DATA_SET}_${MODEL_SIZE}_${SPLIT}_${SUBSET}

RESULT_PATH=${ROOT_PATH}/output/Final/4-All/PRETRAIN_${MODEL_NAME}


# ImageNet finetune
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port ${PORT} main.py \
--eval \
--test-five-crop \
--dataset ${DATA_SET} \
--cfg /mnt/8TDisk1/zhenglab/sunnaizhe/OUC-MOI-ID/SwinTrans-dev/configs/swin/swin_${MODEL_SIZE}_patch4_window7_224.yaml \
--subset ${SUBSET} --split ${SPLIT} \
--batch_size ${BATCH_SIZE} --data-path ${DATA_PATH} \
--resume ${RESULT_PATH}/ckpt_epoch_best.pth \
--output ${RESULT_PATH}/EVAL \
