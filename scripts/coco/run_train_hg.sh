#!/usr/bin/env bash

pushd ../../
source venv/bin/activate

python tools/train.py \
    --cfg experiments/coco/hourglass/hg_template.yaml \
    GPUS '(0,)' \
    DATASET.COLOR_RGB False \
    DATASET.DATASET 'coco' \
    DATASET.ROOT 'data/coco' \
    DATASET.TEST_SET 'val2017' \
    DATASET.TRAIN_SET 'train2017' \
    MODEL.NAME 'hourglass'\
    MODEL.NUM_JOINTS  17 \
    MODEL.INIT_WEIGHTS False \
    MODEL.IMAGE_SIZE 192,256 `# 288,384` \
    MODEL.HEATMAP_SIZE 48,64 `# 72,96` \
    MODEL.SIGMA 2 `# 3` \
    MODEL.EXTRA.NUM_FEATURES  256 \
    MODEL.EXTRA.NUM_STACKS 4 \
    MODEL.EXTRA.NUM_BLOCKS 1 \
    TRAIN.BATCH_SIZE_PER_GPU 4 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 140 \
    TEST.BATCH_SIZE_PER_GPU 32 \
    TEST.USE_GT_BBOX True \
    DEBUG.DEBUG False

deactivate

popd