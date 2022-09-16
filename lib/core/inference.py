# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds

import torch
from torch.nn import functional as F


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def get_final_preds_reg(config, batch_heatmaps, center, scale, beta=1.0):
    _, maxvals = get_max_preds(batch_heatmaps)

    batch_heatmaps = torch.Tensor(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    batch_heatmaps = batch_heatmaps.reshape((batch_heatmaps.shape[0], 16, -1))
    
    # print("accurate beta", beta)
    batch_heatmaps = F.softmax(batch_heatmaps * beta, 2)
    batch_heatmaps = batch_heatmaps.reshape((batch_heatmaps.shape[0], 16, 64, 64))

    accu_x = batch_heatmaps.sum(dim=2)
    accu_y = batch_heatmaps.sum(dim=3)

    accu_x = accu_x * torch.arange(heatmap_width)
    accu_y = accu_y * torch.arange(heatmap_height)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    pred = torch.cat((accu_x, accu_y), dim=2)
    coords = pred.detach().cpu().numpy()

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def get_final_preds_reg_bias(config, batch_heatmaps, center, scale, beta=1.0):
    _, maxvals = get_max_preds(batch_heatmaps)

    batch_heatmaps = torch.Tensor(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    batch_heatmaps = batch_heatmaps.reshape((batch_heatmaps.shape[0], 16, -1))

    ################################################################
    output2 = (batch_heatmaps * beta).detach().cpu().numpy()
    # output2 = output2 - np.max(output2, axis=-1, keepdims=True)
    numerator = np.exp(output2)
    C = np.sum(numerator, axis=-1, keepdims=True)
    A = C / (C - heatmap_width * heatmap_height)
    Bx = (heatmap_height * heatmap_width * heatmap_width) / (2 * (C - heatmap_width * heatmap_height))
    By = (heatmap_height * heatmap_height * heatmap_width) / (2 * (C - heatmap_width * heatmap_height))
    A = torch.Tensor(A)
    Bx = torch.Tensor(Bx)
    By = torch.Tensor(By)
    ################################################################
    
    # print("accurate beta", beta)
    batch_heatmaps = F.softmax(batch_heatmaps * beta, 2)
    batch_heatmaps = batch_heatmaps.reshape((batch_heatmaps.shape[0], 16, 64, 64))

    accu_x = batch_heatmaps.sum(dim=2)
    accu_y = batch_heatmaps.sum(dim=3)

    accu_x = accu_x * torch.arange(heatmap_width)
    accu_y = accu_y * torch.arange(heatmap_height)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    #####################################################################
    # print("x", x.shape)
    accu_x = A.type(torch.FloatTensor) * accu_x - Bx.type(torch.FloatTensor)
    accu_y = A.type(torch.FloatTensor) * accu_y - By.type(torch.FloatTensor)
    #####################################################################

    # accu_x = accu_x / float(heatmap_width) - 0.5
    # accu_y = accu_y / float(heatmap_height) - 0.5

    pred = torch.cat((accu_x, accu_y), dim=2)
    coords = pred.detach().cpu().numpy()

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals