# Complement Objective Training


## Overview

This repository contains the TensorFlow implementation of Complement Objective Training introduced in the following paper:

> _COMPLEMENT OBJECTIVE TRAINING_. <br>
**Hao-Yun Chen**, Pei-Hsin Wang, Chun-Hao Liu, Shih-Chieh Chang, Jia-Yu Pan, Yu-Ting Chen, Wei Wei, Da-Cheng Juan. <br> <https://openreview.net/forum?id=HyM7AiA5YX>


## Dependencies

* Python 3.5 
* tensorflow 1.13.2
* Pillow

Please find details dependency in `requirements.txt`

## Usage

### Data Preparation & Pre-processing

To prepare dataset, please find the folder structure below:

    <path to data dir>
    ├── test
    └── train
        ├── 14
        │   ├── test_14_Left.tif
        │   ├── test_14_Right.tif
        │   └── test_14_Top.tif
        ├── 255
        │   ├── test_255__left.tiff
        │   ├── test_255_right.tiff
        │   └── test_255_top.tiff
        ├── <label id in int>
        │   ├── <images of label id>.tiff
        │   ├  ..........
        ├   ...........
        ├── 37
        │   ├── test_37_left.tiff
        │   ├── test_37_right.tiff
        │   └── test_37_top.tiff
        └── 76
            ├── test_76_left.tif
            ├── test_76_right.tif
            └── test_76_top.tif

To convert dataset into tfrecords please find the following commands:

        python sem_to_tfrecord.py \
          --raw_data_dir=${DATA_DIR}

### Run
For getting baseline results
	
        python main.py \
          --data_dir=${DATA_DIR} \
          --model_dir=${TRAIN_DIR} \
          --train_epochs=200 \
          --epochs_per_eval=10 \
          --batch_size=128 \
          --resnet_size=101
	
For training via Complement objective

        python main.py \
          --data_dir=${DATA_DIR} \
          --model_dir=${TRAIN_DIR} \
          --train_epochs=200 \
          --epochs_per_eval=10 \
          --batch_size=128 \
          --COT \
          --resnet_size=101

Please find the detail runscript under `run.bash`

## Benchmark on CIFAR10

The following table shows the best test accuracy in a 200-epoch training session.

| Model         | Baseline      | COT           |
|:--------------|:--------------|:--------------|
| ResNet-110    | 93.68%        | 93.49%        |


## Acknowledgement
The CIFAR-10 reimplementation of COT is adapted from the [tensorflow/models](https://github.com/tensorflow/models/tree/v1.13.0/official/resnet) repository by [tensorflow](https://github.com/tensorflow).

