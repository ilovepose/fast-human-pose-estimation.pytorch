# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        # print("output", output.shape)  # torch.Size([batch_size, 16, 64, 64])
        # print("target", target.shape)  # torch.Size([batch_size, 16, 64, 64])
        # print("target_weight", target_weight.shape)  # torch.Size([batch_size, 16, 1])
        
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        # print("loss / num_joints", loss / num_joints)  # tensor(0.0835, device='cuda:0', grad_fn=<DivBackward0>) tensor(0.0017, device='cuda:0', grad_fn=<DivBackward0>)

        return loss / num_joints


# 试试OHKM，看看参数量
class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=14):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        # print("self.ohkm(loss)", self.ohkm(loss))

        return self.ohkm(loss)


# class L1JointLocationLoss(nn.Module):
#     def __init__(self, use_target_weight, size_average=True):
#         super(L1JointLocationLoss, self).__init__()
#         self.size_average = size_average
#         self.use_target_weight = use_target_weight

#     # def forward(self, preds, *args):
#     def forward(self, output, target, target_weight):
#         # gt_joints = args[0]
#         # gt_joints_vis = args[1]

#         # num_joints = int(gt_joints_vis.shape[1] / 3)
#         # hm_width = preds.shape[-1]
#         # hm_height = preds.shape[-2]
#         # hm_depth = preds.shape[-3] // num_joints if self.output_3d else 1

#         # pred_jts = softmax_integral_tensor(preds, num_joints, self.output_3d, hm_width, hm_height, hm_depth)

#         # print("output", output.shape)  # torch.Size([batch_size, 16, 64, 64])
#         # print("target", target.shape)  # torch.Size([batch_size, 16, 64, 64])
#         # print("target_weight", target_weight.shape)  # torch.Size([batch_size, 16, 1])

#         gt_joints = target
#         gt_joints_vis = target_weight

#         num_joints = output.size(1)
#         hm_height = output.size(2)
#         hm_width = output.size(3)

#         pred_jts = softmax_integral_tensor(output, num_joints, hm_width, hm_height)

#         _assert_no_grad(gt_joints)
#         _assert_no_grad(gt_joints_vis)

#         # print("pred_jts", pred_jts.shape)  # torch.Size([batch_size, 32])
#         # print("gt_joints", gt_joints.shape)  # torch.Size([batch_size, 16, 64, 64]) => torch.Size([batch_size, 32])
#         # print("pred_jts.shape[0]", pred_jts.shape[0])
#         # print("pred_jts", pred_jts)
#         # print("gt_joints", gt_joints)
#         # print("target_weight", target_weight)

#         out = torch.abs(pred_jts - gt_joints)
        
#         # print("out", out)
#         # print("out.sum() / pred_jts.shape[0]", out.sum() / pred_jts.shape[0])
#         # print("out.mean() / pred_jts.shape[0]", out.mean() / pred_jts.shape[0])
        
#         if self.use_target_weight:
#             # print("gt_joints_vis", gt_joints_vis.shape) # torch.Size([batch_size, 16, 1])
#             gt_joints_vis = gt_joints_vis.squeeze(2)
            
#             # gt_joints_vis = gt_joints_vis.repeat(1,2)

#             # print("gt_joints_vis1", gt_joints_vis)
#             # print("gt_joints_vis2", gt_joints_vis.shape)  # torch.Size([batch_size, 16])
            
#             # # gt_joints_vis = torch.cat((gt_joints_vis, gt_joints_vis), dim=0)
#             # gt_joints_vis = gt_joints_vis.expand(gt_joints_vis.shape[0] * 2, gt_joints_vis.shape[1])
#             # gt_joints_vis = gt_joints_vis.permute(1,0)
            
#             # print("gt_joints_vis3", gt_joints_vis)
#             # print("gt_joints_vis4", gt_joints_vis.shape)  # torch.Size([batch_size*2, 16])
            
#             # gt_joints_vis = gt_joints_vis.reshape((out.shape[0], out.shape[1]))
            
#             # print("gt_joints_vis5", gt_joints_vis)
#             # print("gt_joints_vis6", gt_joints_vis.shape)  # torch.Size([batch_size, 32])

#             gt_joints_vis = torch.repeat_interleave(gt_joints_vis, repeats=2, dim=1)

#             # print("gt_joints_vis", gt_joints_vis)
#             # print("gt_joints_vis.shape", gt_joints_vis.shape)  # torch.Size([batch_size, 32])

#             # print("out", out.shape)  # torch.Size([batch_size, 32])
#             out = out * gt_joints_vis

#         # print("out", out)
#         # print("out.sum() / pred_jts.shape[0]", out.sum() / pred_jts.shape[0])
#         # print("out.mean() / pred_jts.shape[0]", out.mean() / pred_jts.shape[0])
        
#         if self.size_average:
#             return out.mean() / pred_jts.shape[0]
#         else:
#             return out.sum()


# def _assert_no_grad(tensor):
#     assert not tensor.requires_grad, \
#         "nn criterions don't compute the gradient w.r.t. targets - please " \
#         "mark these tensors as not requiring gradients"

# def softmax_integral_tensor(preds, num_joints, hm_width, hm_height):
#     # global soft max
#     preds = preds.reshape((preds.shape[0], num_joints, -1))
#     preds = F.softmax(preds, 2)

#     # print("preds", preds.shape)  # torch.Size([batch_size, 16, 4096])

#     # integrate heatmap into joint location
#     x, y = generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)

#     # x = x / float(hm_width) - 0.5
#     # y = y / float(hm_height) - 0.5

#     preds = torch.cat((x, y), dim=2)
#     # print("preds", preds)
#     # preds = torch.cat((x, y), dim=1)
#     # print("preds", preds)
#     # print("preds", preds.shape)  # torch.Size([batch_size, 16, 2])
#     preds = preds.reshape((preds.shape[0], num_joints * 2))
#     # print("preds", preds.shape)  # torch.Size([batch_size, 32])
#     return preds

# def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
#     assert isinstance(heatmaps, torch.Tensor)

#     heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

#     # print("heatmaps", heatmaps.shape)  # torch.Size([batch_size, 16, 64, 64])

#     accu_x = heatmaps.sum(dim=2)
#     accu_y = heatmaps.sum(dim=3)

#     # print("accu_x", accu_x.shape)  # torch.Size([batch_size, 16, 64])

#     # accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor), devices=[accu_x.device.index])[0]
#     # accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor), devices=[accu_y.device.index])[0]

#     device = torch.device('cuda')

#     accu_x = accu_x * torch.arange(float(x_dim)).to(device)
#     accu_y = accu_y * torch.arange(float(y_dim)).to(device)

#     accu_x = accu_x.sum(dim=2, keepdim=True)
#     accu_y = accu_y.sum(dim=2, keepdim=True)

#     # print("accu_x", accu_x.shape)  # torch.Size([batch_size, 16, 1])

#     return accu_x, accu_y


# class L1JointLocationLoss(nn.Module):
#     def __init__(self, use_target_weight=True, size_average=True):
#         super(L1JointLocationLoss, self).__init__()
#         self.size_average = size_average
#         self.use_target_weight = use_target_weight

#     def forward(self, output, target, target_weight):
#         # print("start")
#         # print("output", output.shape)  # torch.Size([batch, 16, 64, 64])
#         # print("target", target.shape)  # torch.Size([batch, 32])
#         # print("target_weight", target_weight.shape)  # torch.Size([batch, 16, 1])
        
#         gt_joints = target
#         print("gt_joints", gt_joints[0])
#         # print("(gt_joints[0] + 0.5) * 256", (gt_joints[0] + 0.5) * 256)

#         gt_joints_vis = torch.where((gt_joints + 0.5) * 256 < 0, 0.0, 1.0)
#         # print("gt_joints_vis", gt_joints_vis.shape)  # torch.Size([batch, 32])
#         # print("gt_joints_vis", gt_joints_vis[0])
#         # gt_joints_vis = target_weight * target_weight2
        
#         # gt_joints_vis = gt_joints_vis.squeeze(2)
#         # gt_joints_vis = torch.repeat_interleave(gt_joints_vis, repeats=2, dim=1)
#         # print("gt_joints_vis", gt_joints_vis[0])
#         # print("gt_joints_vis", gt_joints_vis.shape)  # torch.Size([batch, 32])

#         num_joints = output.shape[1]
#         hm_height = output.shape[2]
#         hm_width = output.shape[3]

#         output = output.reshape((output.shape[0], num_joints, -1))
        
#         # import pandas as pd
#         # pd.set_option('display.max_columns', None)

#         # print("output", pd.Series(output[0][0].detach().cpu().numpy()))
#         output = F.softmax(output, 2)
#         # print("output2", pd.Series(output[0][0].detach().cpu().numpy()))
#         output = output.reshape((output.shape[0], num_joints, hm_width, hm_height))  # torch.Size([batch, 16, 64, 64])

#         x = output.sum(dim=2)
#         y = output.sum(dim=3)
#         # print("x", x[0,0,:], x.shape)  # torch.Size([batch, 16, 64])
#         # print("y", y[0,0,:], y.shape)
#         # print("x.sum(dim=2)", x.sum(dim=2))
        
#         x = x * torch.arange(hm_width).type(torch.cuda.FloatTensor)
#         y = y * torch.arange(hm_height).type(torch.cuda.FloatTensor)

#         #######################################
#         # device = torch.device('cuda')
#         # x = x * torch.arange(float(hm_width)).to(device)
#         # y = y * torch.arange(float(hm_height)).to(device)
#         #######################################

#         # print("x", x[0,0,:], x.shape)  # torch.Size([batch, 16, 64]) 
#         # print("y", y[0,0,:], y.shape)

#         # x = x * torch.arange(float(hm_width))
#         # y = y * torch.arange(float(hm_height))
#         # print("x", x[0,0,:], x.shape)
        
#         x = x.sum(dim=2, keepdim=True)
#         y = y.sum(dim=2, keepdim=True)
#         # print("x", x, x.shape)  # torch.Size([batch, 16, 1])
#         print("x", x[0,:,0])
#         # print("y", y[0,:,0])

#         x = x / float(hm_width) - 0.5
#         y = y / float(hm_height) - 0.5

#         pred_jts = torch.cat((x, y), dim=2)
#         pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 2))
#         # print("pred_jts", pred_jts[0])
        
#         # for i in range(x.shape[0]):
#         #     l1 = x[i,:,0].detach().cpu().numpy().tolist()
#         #     l2 = y[i,:,0].detach().cpu().numpy().tolist()
#         #     pred_jts[i,:] = [x for y in zip(l1, l2) for x in y]
#         # pred_jts = torch.Tensor(pred_jts).to(device)
#         # # pred_jts = np.dstack((x.detach().cpu().numpy(),y.detach().cpu().numpy())).flatten()
#         # print("pred_jts", pred_jts.shape)
#         # pred_jts = torch.Tensor(pred_jts).to(device)
#         # pred_jts = pred_jts.reshape(-1, num_joints * 2)

#         print("pred_jts", pred_jts[0])
#         print("gt_joints", gt_joints[0])
#         print("gt_joints_vis", gt_joints_vis[0])

#         out = torch.abs(pred_jts - gt_joints)
#         print("out", out[0])
        
#         if self.use_target_weight:
#             out = out * gt_joints_vis
#             print("out * gt_joints_vis", out[0])
        
#         if self.size_average:
#             return out.sum() / (pred_jts.shape[0])
#         else:
#             return out.sum()


# class L1JointLocationLoss(nn.Module):
#     def __init__(self, use_target_weight=True, size_average=True):
#         super(L1JointLocationLoss, self).__init__()
#         self.size_average = size_average
#         self.use_target_weight = use_target_weight

#     def forward(self, output, target, target_weight):      
#         gt_joints = target
#         gt_joints_vis = torch.where((gt_joints + 0.5) * 256 < 1, 0.0, 1.0)

#         num_joints = output.shape[1]
#         hm_height = output.shape[2]
#         hm_width = output.shape[3]

#         output = output.reshape((output.shape[0], num_joints, -1))
#         # print("output", output[0][0].detach().cpu().numpy())

#         alpha = 1.0
#         # output = F.log_softmax(output, 2)
#         output = F.softmax(output * alpha, dim=2)
#         # print("output2", output[0][0].detach().cpu().numpy())
#         output = output.reshape((output.shape[0], num_joints, hm_width, hm_height))

#         x = output.sum(dim=2)
#         y = output.sum(dim=3)
#         # print("x.sum(dim=2)", x.sum(dim=2))
        
#         x = x * torch.arange(hm_width).type(torch.cuda.FloatTensor)
#         y = y * torch.arange(hm_height).type(torch.cuda.FloatTensor)

#         x = x.sum(dim=2, keepdim=True)
#         y = y.sum(dim=2, keepdim=True)

#         x = x / float(hm_width) - 0.5
#         y = y / float(hm_height) - 0.5

#         pred_jts = torch.cat((x, y), dim=2)
#         pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 2))

#         out = torch.abs(pred_jts - gt_joints)
#         # print("pred_jts", pred_jts[0])
#         # print("gt_joints", gt_joints[0])
        
#         if self.use_target_weight:
#             out = out * gt_joints_vis
#             # print("out", out[0])
        
#         if self.size_average:
#             return out.sum() / (pred_jts.shape[0])
#         else:
#             return out.sum()


class L1JointLocationLoss(nn.Module):
    def __init__(self, use_target_weight=True, size_average=True, beta=1.0):
        super(L1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.beta = beta

    def forward(self, output, target, target_weight):      
        gt_joints = target
        gt_joints_vis = torch.where((gt_joints + 0.5) * 256 < 1, 0.0, 1.0)

        num_joints = output.shape[1]
        hm_height = output.shape[2]
        hm_width = output.shape[3]

        output = output.reshape((output.shape[0], num_joints, -1))

        # print("loss beta", self.beta)
        output = F.softmax(output * self.beta, dim=2)
        output = output.reshape((output.shape[0], num_joints, hm_width, hm_height))

        x = output.sum(dim=2)
        y = output.sum(dim=3)
        
        x = x * torch.arange(hm_width).type(torch.cuda.FloatTensor)
        y = y * torch.arange(hm_height).type(torch.cuda.FloatTensor)

        x = x.sum(dim=2, keepdim=True)
        y = y.sum(dim=2, keepdim=True)

        x = x / float(hm_width) - 0.5
        y = y / float(hm_height) - 0.5

        pred_jts = torch.cat((x, y), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 2))

        out = torch.abs(pred_jts - gt_joints)

        if self.use_target_weight:
            out = out * gt_joints_vis
        
        # print("pred_jts", pred_jts[0])
        # print("gt_joints", gt_joints[0])
        # print("out", out[0])

        if self.size_average:
            return out.sum() / (pred_jts.shape[0] * num_joints)
        else:
            return out.sum()


# class L2JointLocationLoss(nn.Module):
#     def __init__(self, use_target_weight, size_average=True):
#         super(L2JointLocationLoss, self).__init__()
#         self.size_average = size_average
#         self.use_target_weight = use_target_weight

#     def forward(self, output, target, target_weight):
#         gt_joints = target
#         gt_joints_vis = target_weight

#         num_joints = output.size(1)
#         hm_height = output.size(2)
#         hm_width = output.size(3)

#         output = output.reshape((output.shape[0], num_joints, -1))
#         output = F.softmax(output, 2)
#         output = output.reshape((output.shape[0], num_joints, hm_width, hm_height))

#         x = output.sum(dim=2)
#         y = output.sum(dim=3)
        
#         device = torch.device('cuda')
#         x = x * torch.arange(float(x)).to(device)
#         y = y * torch.arange(float(y)).to(device)
        
#         x = x.sum(dim=2, keepdim=True)
#         y = y.sum(dim=2, keepdim=True)

#         pred_jts = torch.cat((x, y), dim=2)
#         pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 2))

#         out = (pred_jts - gt_joints) ** 2
        
#         if self.use_target_weight:
#             gt_joints_vis = gt_joints_vis.squeeze(2)
#             gt_joints_vis = torch.repeat_interleave(gt_joints_vis, repeats=2, dim=1)
#             out = out * gt_joints_vis
        
#         if self.size_average:
#             return out.sum() / pred_jts.shape[0]
#         else:
#             return out.sum()


class L2JointLocationLoss(nn.Module):
    def __init__(self, use_target_weight=True, size_average=True, beta=1.0):
        super(L2JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.beta = beta

    def forward(self, output, target, target_weight):      
        gt_joints = target
        gt_joints_vis = torch.where((gt_joints + 0.5) * 256 < 1, 0.0, 1.0)

        num_joints = output.shape[1]
        hm_height = output.shape[2]
        hm_width = output.shape[3]

        output = output.reshape((output.shape[0], num_joints, -1))

        output = F.softmax(output * self.beta, dim=2)
        output = output.reshape((output.shape[0], num_joints, hm_width, hm_height))

        x = output.sum(dim=2)
        y = output.sum(dim=3)
        
        x = x * torch.arange(hm_width).type(torch.cuda.FloatTensor)
        y = y * torch.arange(hm_height).type(torch.cuda.FloatTensor)

        x = x.sum(dim=2, keepdim=True)
        y = y.sum(dim=2, keepdim=True)

        x = x / float(hm_width) - 0.5
        y = y / float(hm_height) - 0.5

        pred_jts = torch.cat((x, y), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 2))

        out = (pred_jts - gt_joints) ** 2
        
        if self.use_target_weight:
            out = out * gt_joints_vis
        
        if self.size_average:
            return out.sum() / (pred_jts.shape[0] * num_joints)
        else:
            return out.sum()


class L1BiasLoss(nn.Module):
    def __init__(self, use_target_weight=True, size_average=True, beta=1.0):
        super(L1BiasLoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.beta = beta

    def forward(self, output, target, target_weight):      
        gt_joints = target
        gt_joints_vis = torch.where((gt_joints + 0.5) * 256 < 1, 0.0, 1.0)

        num_joints = output.shape[1]
        hm_height = output.shape[2]
        hm_width = output.shape[3]

        output = output.reshape((output.shape[0], num_joints, -1))

        # print("loss beta", self.beta)
        
        ################################################################
        output2 = (output * self.beta).detach().cpu().numpy()
        # z = output - np.max(output, axis=-1, keepdims=True)
        numerator = np.exp(output2)
        C = np.sum(numerator, axis=-1, keepdims=True)
        # output = numerator / C
        A = C / (C - hm_width * hm_height)
        Bx = (hm_height * hm_width * hm_width) / (2 * (C - hm_width * hm_height))
        By = (hm_height * hm_height * hm_width) / (2 * (C - hm_width * hm_height))
        A = torch.Tensor(A)
        Bx = torch.Tensor(Bx)
        By = torch.Tensor(By)
        # output = A * output - B
        # print("C", C, C.shape)
        # print("AB", A.shape, B.shape)
        ################################################################

        output = F.softmax(output * self.beta, dim=2)
        output = output.reshape((output.shape[0], num_joints, hm_width, hm_height))

        x = output.sum(dim=2)
        y = output.sum(dim=3)
        
        x = x * torch.arange(hm_width).type(torch.cuda.FloatTensor)
        y = y * torch.arange(hm_height).type(torch.cuda.FloatTensor)

        x = x.sum(dim=2, keepdim=True)
        y = y.sum(dim=2, keepdim=True)

        #####################################################################
        # print("x", x.shape)
        x = A.type(torch.cuda.FloatTensor) * x - Bx.type(torch.cuda.FloatTensor)
        y = A.type(torch.cuda.FloatTensor) * y - By.type(torch.cuda.FloatTensor)
        #####################################################################

        # 归一化
        x = x / float(hm_width) - 0.5
        y = y / float(hm_height) - 0.5

        pred_jts = torch.cat((x, y), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 2))

        out = torch.abs(pred_jts - gt_joints)

        if self.use_target_weight:
            out = out * gt_joints_vis
        
        # print("pred_jts", pred_jts[0])
        # print("gt_joints", gt_joints[0])
        # print("out", out[0])

        if self.size_average:
            return out.sum() / (pred_jts.shape[0] * num_joints)
        else:
            return out.sum()


class FocalL2Loss(nn.Module):
    """
    Compute focal l2 loss between predict and groundtruth
    :param thre: the threshold to distinguish between the foreground
                 heatmap pixels and the background heatmap pixels
    :param alpha beta: compensation factors to reduce the punishment of easy
                 samples (both easy foreground pixels and easy background pixels) 
    """
    def __init__(self, use_target_weight, thre=0.01, alpha=0.1, beta=0.02):
        super().__init__()
        self.thre = thre
        self.alpha = alpha
        self.beta = beta
        self.use_target_weight = use_target_weight
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self, pred, gt, mask):
        """Forward function.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred (torch.Tensor[N,K,H,W]):heatmap of output.
            gt (torch.Tensor[N,K,H,W]): target heatmap.
            mask (torch.Tensor[N,H,W]): mask of target.
        """
        assert pred.size() == gt.size()
        st = torch.where(torch.ge(gt, self.thre), pred - self.alpha, 1 - pred - self.beta)
        factor = torch.abs(1. - st)
        # print("factor", factor.shape)  # torch.Size([1, 16, 64, 64])

        if self.use_target_weight:
            target_weight = mask.unsqueeze(-1).expand_as(pred)
            # print("target_weight", target_weight.shape)  # torch.Size([1, 16, 64, 64])
            loss = ((pred - gt)**2 * factor) * target_weight
            # print("loss", loss.shape)  # torch.Size([1, 16, 64, 64])
        else:
            loss = (pred - gt)**2 * factor
        
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        
        # print("loss", loss)  # tensor([0.1648], device='cuda:0', grad_fn=<MeanBackward1>) tensor([0.0118], device='cuda:0', grad_fn=<MeanBackward1>)
        # tensor(0.1079, device='cuda:0', grad_fn=<MeanBackward1>)

        return loss