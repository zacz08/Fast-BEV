#!/usr/bin/env bash
T=`date +%m%d%H%M`

WORK_DIR=${WORK_DIR:-/home/zc/Fast-BEV}
echo work_dirs: $WORK_DIR

START_TIME=`date +%Y%m%d-%H:%M:%S`

function train {
    GPUS=$1
    EXPNAME=$2
    CONFIG=${3:-$WORK_DIR/configs/fastbev/exp/$EXPNAME.py}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
    NNODES=`expr $GPUS / $GPUS_PER_NODE`
    LOG_FILE=arun_log/paper_train_${START_TIME}.log

    MASTER_PORT=${MASTER_PORT:-28596}
    MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    RANK=${RANK:-0}

    PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        --nnodes=${NNODES} \
        --node_rank=${RANK} \
        $(dirname "$0")/train.py \
        $CONFIG \
        --launcher="pytorch" ${@:4} \
        --work-dir /home/zc/Fast-BEV/train_result \
        # 2>&1 | tee $LOG_FILE > /dev/null &
        # 2>&1 | tee $LOG_FILE
}


function test {
    GPUS=$1
    EXPNAME=$2
    CONFIG=${3:-$WORK_DIR/configs/fastbev/exp/$EXPNAME.py}
    CKPT=${4:-$WORK_DIR/exp/$EXPNAME/epoch_20.pth}
    RESULT=${5:-$WORK_DIR/exp/$EXPNAME/results/results.pkl}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo test; sleep 0.5s
    echo CONFIG: $CONFIG

    PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        $(dirname "$0")/test.py \
        $CONFIG \
        $CKPT \
        --launcher="pytorch" ${@:4} \
        --out $RESULT \
        --eval bbox \
        # 2>&1 | tee /home/zc/Fast-BEV/exp/$EXPNAME/log.eval.$T.txt > /dev/null &
        2>&1 | tee /home/zc/Fast-BEV/exp/$EXPNAME/log.test.$T.txt
}

function evaluation {
    GPUS=$1
    EXPNAME=$2
    CONFIG=${3:-$WORK_DIR/configs/fastbev/exp/$EXPNAME.py}
    RESULT=${4:-$WORK_DIR/exp/$EXPNAME/results/results.pkl}
    GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

    echo evaluation; sleep 0.5s
    echo CONFIG: $CONFIG

    PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        $(dirname "$0")/eval.py \
        $CONFIG \
        --launcher="pytorch" ${@:4} \
        --out $RESULT \
        --eval bbox \
        # 2>&1 | tee /home/zc/Fast-BEV/exp/$EXPNAME/log.eval.$T.txt > /dev/null &
        2>&1 | tee /home/zc/Fast-BEV/exp/$EXPNAME/log.eval.$T.txt
}

# train
train 1 paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4

# test
# test 2 paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4
# test 2 paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4


# eval
# evaluation 1 paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4
# slurm_eval 1 paper/fastbev_m1_r18_s320x880_v200x200x4_c192_d2_f4
# slurm_eval 1 paper/fastbev_m2_r34_s256x704_v200x200x4_c224_d4_f4
# slurm_eval 1 paper/fastbev_m4_r50_s320x880_v250x250x6_c256_d6_f4
# slurm_eval 1 paper/fastbev_m5_r50_s512x1408_v250x250x6_c256_d6_f4
