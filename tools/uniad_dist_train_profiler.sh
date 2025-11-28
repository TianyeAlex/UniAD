#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

export TORCH_CUDA_ARCH_LIST=9.0
export LD_LIBRARY_PATH=/root/anaconda3/envs/uniad_new/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH
export PATH=/root/cuda-12.2/bin/:$PATH
export LD_LIBRARY_PATH=/root/cuda-12.2/lib64/:$LD_LIBRARY_PATH14
export CUDA_HOME=/root/cuda-12.2
source /root/anaconda3/etc/profile.d/conda.sh
conda activate uniad_new
export NCCL_IB_GID_INDEX="3"
sed -i '1s|#!/ssd2/wenshengzhao/anaconda3/envs/uniad_new/bin/python|#!/root/anaconda3/envs/uniad_new/bin/python|' /root/anaconda3/envs/uniad_new/bin/torchrun

MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}  
RANK=${RANK:-0}

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
    --nnodes=${NNODES} \
    --node_rank=${RANK} \
    $(dirname "$0")/profiler_train.py \
    $CFG \
    --profiler-export tensorboard \
    --profiler-no-shapes \
    --profiler-no-stack \
    --profiler-no-flops \
    --launcher pytorch ${@:3} \
    --deterministic \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T