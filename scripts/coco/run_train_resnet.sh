#!/usr/bin/env bash

pushd ../../
source venv/bin/activate

python tools/train.py \
    --cfg experiments/coco/resnet/resnet_template.yaml \
    GPUS '(0,)' \
    DATASET.DATASET 'coco' \
    DATASET.ROOT 'data/coco' \
    DATASET.TEST_SET 'val2017' \
    DATASET.TRAIN_SET 'train2017' \
    MODEL.NAME 'pose_resnet'\
    MODEL.NUM_JOINTS 17 \
    MODEL.INIT_WEIGHTS True \
    MODEL.PRETRAINED 'models/pytorch/imagenet/resnet50-19c8e357.pth' `# resnet101-5d3b4d8f.pth, resnet152-b121ed2d.pth ` \
    MODEL.IMAGE_SIZE 192,256 `# 288,384` \
    MODEL.HEATMAP_SIZE 48,64 `# 72,96` \
    MODEL.SIGMA 2 `# 3` \
    MODEL.EXTRA.NUM_LAYERS 50 `# 101 or 152` \
    TRAIN.BATCH_SIZE_PER_GPU 4 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 140 \
    TEST.BATCH_SIZE_PER_GPU 32 `# 24` \
    TEST.USE_GT_BBOX True \
    DEBUG.DEBUG False

deactivate

popd