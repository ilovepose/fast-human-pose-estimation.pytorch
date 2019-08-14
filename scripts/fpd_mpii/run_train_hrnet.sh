#!/usr/bin/env bash

pushd ../../
source venv/bin/activate

python tools/fpd_train.py \
    --tcfg experiments/fpd_mpii/hrnet/w48_256x256_adam_lr1e-3.yaml `# experiments/fpd/hrnet/w48_256x192_adam_lr1e-3.yaml`\
    --cfg experiments/fpd_mpii/hrnet/hrnet_template.yaml \
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
    TRAIN.BATCH_SIZE_PER_GPU 16 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 20 \
    TRAIN.LR 0.00001 \
    TRAIN.LR_STEP 5,10,15 \
    TRAIN.CHECKPOINT 'models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth' \
    TEST.BATCH_SIZE_PER_GPU 32 \
    DEBUG.DEBUG False \
    KD.TRAIN_TYPE 'FPD' `#FPD`\
    KD.TEACHER 'models/pytorch/pose_mpii/pose_hrnet_w48_256x256.pth' \
    KD.ALPHA 0.5 \

deactivate

popd