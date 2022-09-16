# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cgi import print_directory

import numpy as np

from core.inference import get_max_preds
import torch
from torch.nn import functional as F


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


# def accuracy_reg(output, target):
#     '''
#     Calculate accuracy according to PCK,
#     but uses ground truth heatmap rather than x,y locations
#     First value to be returned is average accuracy across 'idxs',
#     followed by individual accuracies
#     '''
#     idx = list(range(output.shape[1]))
#     norm = 1.0

#     # pred, _ = get_max_preds(output)
#     # print("output", output.shape)  # torch.Size([1, 16, 64, 64])

#     output = output.reshape((output.shape[0], 16, -1))
#     # print("output", output.shape)
#     output = F.softmax(output, 2)
#     # print("output", output.shape)

#     output = output.reshape((output.shape[0], 16, 64, 64))
#     # print("output", output.shape)
#     accu_x = output.sum(dim=2)
#     accu_y = output.sum(dim=3)

#     device = torch.device('cuda')

#     accu_x = accu_x * torch.arange(float(64)).to(device)
#     accu_y = accu_y * torch.arange(float(64)).to(device)

#     accu_x = accu_x.sum(dim=2, keepdim=True)
#     accu_y = accu_y.sum(dim=2, keepdim=True)

#     pred = torch.cat((accu_x, accu_y), dim=2)
#     # print("pred", preds.shape)  # torch.Size([1, 16, 2])
#     # pred = pred.reshape((preds.shape[0], 16 * 2))
#     # print("pred", pred.shape)  # torch.Size([1, 32])
#     pred = pred.detach().cpu().numpy()
#     # print("pred", pred.shape)  # (1, 16, 2)

#     # print("target", target.shape)  # (1, 16, 64, 64)
#     target, _ = get_max_preds(target)
#     # print("target", target.shape)  # (1, 16, 2)

#     h = output.shape[2]
#     w = output.shape[3]
#     norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    
#     dists = calc_dists(pred, target, norm)

#     acc = np.zeros((len(idx) + 1))
#     avg_acc = 0
#     cnt = 0

#     for i in range(len(idx)):
#         acc[i + 1] = dist_acc(dists[idx[i]])
#         if acc[i + 1] >= 0:
#             avg_acc = avg_acc + acc[i + 1]
#             cnt += 1

#     avg_acc = avg_acc / cnt if cnt != 0 else 0
#     if cnt != 0:
#         acc[0] = avg_acc
#     return acc, avg_acc, cnt, pred


def accuracy_reg(output, target, beta=1.0):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0

    hm_height = output.shape[2]
    hm_width = output.shape[3]

    output = output.reshape((output.shape[0], 16, -1))
    
    # print("accurate beta", beta)
    output = F.softmax(output * beta, 2)
    output = output.reshape((output.shape[0], 16, 64, 64))

    accu_x = output.sum(dim=2)
    accu_y = output.sum(dim=3)

    accu_x = accu_x * torch.arange(hm_width).type(torch.cuda.FloatTensor)
    accu_y = accu_y * torch.arange(hm_height).type(torch.cuda.FloatTensor)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    pred = torch.cat((accu_x, accu_y), dim=2)
    pred = pred.detach().cpu().numpy()

    target, _ = get_max_preds(target)

    norm = np.ones((pred.shape[0], 2)) * np.array([hm_height, hm_width]) / 10
    
    dists = calc_dists(pred, target, norm)

    # print("pred", pred[0])
    # print("target", target[0])
    # print("norm", norm)
    # print("dists", dists)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def accuracy_reg_bias(output, target, beta=1.0):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0

    hm_height = output.shape[2]
    hm_width = output.shape[3]

    output = output.reshape((output.shape[0], 16, -1))

    ################################################################
    output2 = (output * beta).detach().cpu().numpy()
    # output2 = output2 - np.max(output2, axis=-1, keepdims=True)
    numerator = np.exp(output2)
    C = np.sum(numerator, axis=-1, keepdims=True)
    A = C / (C - hm_width * hm_height)
    Bx = (hm_height * hm_width * hm_width) / (2 * (C - hm_width * hm_height))
    By = (hm_height * hm_height * hm_width) / (2 * (C - hm_width * hm_height))
    A = torch.Tensor(A)
    Bx = torch.Tensor(Bx)
    By = torch.Tensor(By)
    ################################################################
    
    # print("accurate beta", beta)
    output = F.softmax(output * beta, 2)
    output = output.reshape((output.shape[0], 16, 64, 64))

    accu_x = output.sum(dim=2)
    accu_y = output.sum(dim=3)

    accu_x = accu_x * torch.arange(hm_width).type(torch.cuda.FloatTensor)
    accu_y = accu_y * torch.arange(hm_height).type(torch.cuda.FloatTensor)

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    #####################################################################
    # print("x", x.shape)
    accu_x = A.type(torch.cuda.FloatTensor) * accu_x - Bx.type(torch.cuda.FloatTensor)
    accu_y = A.type(torch.cuda.FloatTensor) * accu_y - By.type(torch.cuda.FloatTensor)
    #####################################################################

    # accu_x = accu_x / float(hm_width) - 0.5
    # accu_y = accu_y / float(hm_height) - 0.5

    pred = torch.cat((accu_x, accu_y), dim=2)
    pred = pred.detach().cpu().numpy()

    target, _ = get_max_preds(target)

    norm = np.ones((pred.shape[0], 2)) * np.array([hm_height, hm_width]) / 10
    
    dists = calc_dists(pred, target, norm)

    # print("pred", pred[0])
    # print("target", target[0])
    # print("norm", norm)
    # print("dists", dists)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred