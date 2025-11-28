#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
# -------------------------------------------------- #

# export CUDA_VISIBLE_DEVICES=MIG-0af3be12-7a61-55c8-8f1e-2ae48cdee736,MIG-aadc580e-3cb7-54aa-b737-e9222c354c6a
# export CUDA_VISIBLE_DEVICES=MIG-0af3be12-7a61-55c8-8f1e-2ae48cdee736
export CUDA_VISIBLE_DEVICES=MIG-aadc580e-3cb7-54aa-b737-e9222c354c6a
python -c "import torch; print(torch.cuda.device_count())"

GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
# NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  
# RANK=${RANK:-0}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=2 \
    --node_rank=1 \
    $(dirname "$0")/train.py \
    $CFG \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T