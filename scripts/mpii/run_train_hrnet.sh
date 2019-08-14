#!/usr/bin/env bash

pushd ../../
source venv/bin/activate

python tools/train.py \
    --cfg experiments/mpii/hrnet/hrnet_template.yaml \
    GPUS '(0,)' \
    DATASET.COLOR_RGB True \
    DATASET.NUM_JOINTS_HALF_BODY 8 \
    DATASET.PROB_HALF_BODY -1.0 \
    DATASET.DATASET 'mpii' \
    DATASET.ROOT 'data/mpii' \
    DATASET.TEST_SET 'valid' \
    DATASET.TRAIN_SET 'train' \
    MODEL.NAME 'pose_hrnet'\
    MODEL.NUM_JOINTS 16 \
    MODEL.INIT_WEIGHTS True \
    MODEL.PRETRAINED 'models/pytorch/imagenet/hrnet_w32-36af842e.pth' `# hrnet_w48-8ef0771d.pth` \
    MODEL.IMAGE_SIZE 256,256 \
    MODEL.HEATMAP_SIZE 64,64 \
    MODEL.SIGMA 2 \
    MODEL.EXTRA.STAGE2.NUM_CHANNELS 32,64 `# 48,96` \
    MODEL.EXTRA.STAGE3.NUM_CHANNELS 32,64,128 `# 48,96,192` \
    MODEL.EXTRA.STAGE4.NUM_CHANNELS 32,64,128,256 `# 48,96,192,384` \
    TRAIN.BATCH_SIZE_PER_GPU 4 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 140 \
    TEST.BATCH_SIZE_PER_GPU 32 `# 24` \
    DEBUG.DEBUG False

deactivate

popd