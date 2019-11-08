#!/usr/bin/env bash

pushd ../../
source venv/bin/activate

python tools/fpd_train.py \
    --tcfg experiments/fpd_mpii/hourglass/hg8_256x256_d256x3_adam_lr2.5e-4.yaml \
    --cfg experiments/fpd_mpii/hourglass/hg_template.yaml \
    GPUS '(0,)' \
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
    MODEL.EXTRA.NUM_FEATURES  128 \
    MODEL.EXTRA.NUM_STACKS 4 `# 8` \
    MODEL.EXTRA.NUM_BLOCKS 1 \
    TRAIN.BATCH_SIZE_PER_GPU 4 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 140 \
    TRAIN.LR 0.00025 \
    TRAIN.CHECKPOINT 'models/pytorch/pose_mpii/bs4_hourglass_128_4_1_16_0.00025_0_140_87.934_model_best.pth' `#models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth` \
    TEST.BATCH_SIZE_PER_GPU 32 \
    DEBUG.DEBUG False \
    KD.TRAIN_TYPE 'FPD' `#FPD`\
    KD.TEACHER 'models/pytorch/pose_mpii/bs4_hourglass_256_8_1_16_0.00025_0_140_90.520_model_best.pth' \
    KD.ALPHA 0.5 \

deactivate

popd