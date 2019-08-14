#!/usr/bin/env bash

pushd ../../
source venv/bin/activate

python tools/fpd_train.py \
    --tcfg experiments/fpd_coco/hrnet/w48_256x192_adam_lr1e-3.yaml `# experiments/fpd/hrnet/w48_256x192_adam_lr1e-3.yaml`\
    --cfg experiments/fpd_coco/hrnet/hrnet_template.yaml \
    GPUS '(0,1,2,3)' \
    DATASET.CACHE_ROOT 'data/cache' `# cache point path`\
    DATASET.COLOR_RGB True \
    DATASET.NUM_JOINTS_HALF_BODY 8 \
    DATASET.PROB_HALF_BODY 0.3 \
    DATASET.DATASET 'coco' \
    DATASET.ROOT 'data/coco' \
    DATASET.TEST_SET 'val2017' \
    DATASET.TRAIN_SET 'train2017' \
    MODEL.NAME 'pose_hrnet'\
    MODEL.NUM_JOINTS 17 \
    MODEL.INIT_WEIGHTS True \
    MODEL.PRETRAINED 'models/pytorch/imagenet/hrnet_w32-36af842e.pth' `# hrnet_w48-8ef0771d.pth` \
    MODEL.IMAGE_SIZE 192,256 `# 288,384` \
    MODEL.HEATMAP_SIZE 48,64 `# 72,96` \
    MODEL.SIGMA 2 `# 3` \
    MODEL.EXTRA.STAGE2.NUM_CHANNELS 32,64 `# 48,96` \
    MODEL.EXTRA.STAGE3.NUM_CHANNELS 32,64,128 `# 48,96,192` \
    MODEL.EXTRA.STAGE4.NUM_CHANNELS 32,64,128,256 `# 48,96,192,384` \
    TRAIN.BATCH_SIZE_PER_GPU 24 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 210 \
    TRAIN.LR 0.001 \
    TRAIN.LR_STEP 170,200 \
    TRAIN.CHECKPOINT 'models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth' \
    TEST.BATCH_SIZE_PER_GPU 32 \
    TEST.USE_GT_BBOX False \
    DEBUG.DEBUG False \
    KD.TRAIN_TYPE 'FPD' `#FPD`\
    KD.TEACHER 'models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth' `# models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth`\
    KD.ALPHA 0.5 \

deactivate

popd
