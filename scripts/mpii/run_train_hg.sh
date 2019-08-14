#!/usr/bin/env bash

pushd ../../
source venv/bin/activate

python tools/train.py \
    --cfg experiments/mpii/hourglass/hg_template.yaml \
    GPUS '(0,1)' \
    DATASET.COLOR_RGB False \
    DATASET.DATASET 'mpii' \
    DATASET.ROOT 'data/mpii' \
    DATASET.NUM_JOINTS_HALF_BODY 8 \
    DATASET.PROB_HALF_BODY -1.0 \
    DATASET.TEST_SET 'valid' \
    DATASET.TRAIN_SET 'train' \
    MODEL.NAME 'hourglass'\
    MODEL.NUM_JOINTS  16 \
    MODEL.INIT_WEIGHTS False \
    MODEL.IMAGE_SIZE 256,256 \
    MODEL.HEATMAP_SIZE 64,64 \
    MODEL.SIGMA 2 \
    MODEL.EXTRA.NUM_FEATURES  256 \
    MODEL.EXTRA.NUM_STACKS 8 `# 8` \
    MODEL.EXTRA.NUM_BLOCKS 1 \
    TRAIN.BATCH_SIZE_PER_GPU 4 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 140 \
    TEST.BATCH_SIZE_PER_GPU 32 \
    DEBUG.DEBUG False

deactivate

popd
