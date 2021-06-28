#!/bin/bash -e

export CUDA_VISIBLE_DEVICES=1

function log_print {
  printf "%s.%.6s: %s\n" "`date '+%F %T'`" "`date '+%N'`" "${1}"
}

TRAIN_DATA_SIZE=50000
BATCH_SIZE=128
EPOCH=200
#EPOCH=10
TRAIN_STEPS=`expr $TRAIN_DATA_SIZE / $BATCH_SIZE \* $EPOCH`

cifar10_DATA_DIR="${HOME}/waste/SEM"
cifar10_TRAIN_DIR="${HOME}/cifar10_train"
cifar10_EVAL_DIR="${HOME}/cifar10_eval"

# convert dataset to tfrecords
python3 sem_to_tfrecord.py \
  --raw_data_dir=${cifar10_DATA_DIR}

python3 cifar10_main.py \
  --data_dir=${cifar10_DATA_DIR} \
  --model_dir=${cifar10_TRAIN_DIR} \
  --train_epochs=$EPOCH \
  --epochs_per_eval=10 \
  --batch_size=$BATCH_SIZE \
  -c \
  --resnet_size=101

unset CUDA_VISIBLE_DEVICES
