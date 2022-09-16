# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Written by Feng Zhang & Hong Hu
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy, accuracy_reg, accuracy_reg_bias
from core.inference import get_final_preds, get_final_preds_reg, get_final_preds_reg_bias
from utils.transforms import flip_back
from utils.vis import save_debug_images

from core.inference import get_max_preds


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
            output = outputs[-1]
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def fpd_train(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA

    # s_model switch to train mode and t_model switch to evaluate mode
    model.train()
    tmodel.eval()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        toutput = tmodel(input)
        if isinstance(toutput, list):
            toutput = toutput[-1]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[0], target, target_weight)
            kd_pose_loss = kd_pose_criterion(outputs[0], toutput, target_weight)

            for output in outputs[1:]:
                pose_loss += pose_criterion(output, target, target_weight)
                kd_pose_loss += kd_pose_criterion(output, toutput, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target, target_weight)
            kd_pose_loss = kd_pose_criterion(output, toutput, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(pose_loss.item(), input.size(0))
        kd_pose_losses.update(kd_pose_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def train_reg(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    #########################################
    valid_type = config.KD.VALID_TYPE
    beta = config.LOSS.BETA
    #########################################

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)  # 有几个stack就进行几次中间监督，列表中就有多少个output
        # print("outputs", len(outputs))
        # print("outputs!!!", outputs[0][0][0])  # TODO 所以与softmax有很大的关系

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        ##########################################################################################
        # print("meta['joints']", meta['joints'])
        # print("meta['joints_vis']", meta['joints_vis'])
        # print("meta['joints']", meta['joints'].shape)  # torch.Size([batch_size, 16, 3])
        # print("meta['joints']", meta['joints'][:,:,0:2].shape)  # torch.Size([batch_size, 16, 2])
        target_reg = meta['joints'][:,:,0:2]
        target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
        # print("target_reg[0]1", target_reg[0])
        target_reg = target_reg / 256 - 0.5
        # print("target_reg[0]2", target_reg[0])

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target_reg, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target_reg, target_weight)
            output = outputs[-1]
        else:
            output = outputs
            loss = criterion(output, target_reg, target_weight)
        ##########################################################################################

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        ##########################################################################################
        if valid_type == 'hm':
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
        
        elif valid_type == 'reg':
            _, avg_acc, cnt, pred = accuracy_reg(output, target.detach().cpu().numpy(), beta)
        
        elif valid_type == 'reg_bias':
            _, avg_acc, cnt, pred = accuracy_reg_bias(output, target.detach().cpu().numpy(), beta)

        else:
            raise NotImplementedError
        ##########################################################################################
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def fpd_train_thm_sreg(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA
    #########################################
    valid_type = config.KD.VALID_TYPE
    beta = config.LOSS.BETA
    #########################################

    # s_model switch to train mode and t_model switch to evaluate mode
    model.train()
    tmodel.eval()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        toutput = tmodel(input)
        if isinstance(toutput, list):
            toutput = toutput[-1]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        ##########################################################################################
        target_reg = meta['joints'][:,:,0:2]
        target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
        target_reg = target_reg / 256 - 0.5

        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[0], target_reg, target_weight)
            kd_pose_loss = kd_pose_criterion(outputs[0], toutput, target_weight)

            for output in outputs[1:]:
                pose_loss += pose_criterion(output, target_reg, target_weight)
                kd_pose_loss += kd_pose_criterion(output, toutput, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target_reg, target_weight)
            kd_pose_loss = kd_pose_criterion(output, toutput, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
        ##########################################################################################

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(pose_loss.item(), input.size(0))
        kd_pose_losses.update(kd_pose_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        ##########################################################################################
        if valid_type == 'hm':
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
        
        elif valid_type == 'reg':
            _, avg_acc, cnt, pred = accuracy_reg(output, target.detach().cpu().numpy(), beta)
        
        elif valid_type == 'reg_bias':
            _, avg_acc, cnt, pred = accuracy_reg_bias(output, target.detach().cpu().numpy(), beta)       
        
        else:
            raise NotImplementedError
        ##########################################################################################
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def fpd_train_treg_shm(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA

    # s_model switch to train mode and t_model switch to evaluate mode
    model.train()
    tmodel.eval()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        toutput = tmodel(input)
        if isinstance(toutput, list):
            toutput = toutput[-1]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        ##########################################################################################
        # print("toutput", toutput.shape)  # torch.Size([1, 16, 64, 64])
        toutput_reg, _ = get_max_preds(toutput.detach().cpu().numpy())
        # print("toutput_reg", toutput_reg.shape)  # (1, 16, 2)
        toutput_reg = torch.Tensor(toutput_reg)
        # print("toutput_reg", toutput_reg.shape)  # torch.Size([1, 16, 2])
        toutput_reg = toutput_reg.reshape((toutput_reg.shape[0], 16 * 2)).cuda()
        toutput_reg = toutput_reg / 64 - 0.5

        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[0], target, target_weight)
            kd_pose_loss = kd_pose_criterion(outputs[0], toutput_reg, target_weight)

            for output in outputs[1:]:
                pose_loss += pose_criterion(output, target, target_weight)
                kd_pose_loss += kd_pose_criterion(output, toutput_reg, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target, target_weight)
            kd_pose_loss = kd_pose_criterion(output, toutput_reg, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
        ##########################################################################################

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(pose_loss.item(), input.size(0))
        kd_pose_losses.update(kd_pose_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def fpd_train_treg_sreg(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA
    #########################################
    valid_type = config.KD.VALID_TYPE
    beta = config.LOSS.BETA
    #########################################

    # s_model switch to train mode and t_model switch to evaluate mode
    model.train()
    tmodel.eval()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        toutput = tmodel(input)
        if isinstance(toutput, list):
            toutput = toutput[-1]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        ##########################################################################################
        target_reg = meta['joints'][:,:,0:2]
        target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
        target_reg = target_reg / 256 - 0.5

        toutput_reg, _ = get_max_preds(toutput.detach().cpu().numpy())
        toutput_reg = torch.Tensor(toutput_reg)
        toutput_reg = toutput_reg.reshape((toutput_reg.shape[0], 16 * 2)).cuda()
        toutput_reg = toutput_reg / 64 - 0.5

        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[0], target_reg, target_weight)
            kd_pose_loss = kd_pose_criterion(outputs[0], toutput_reg, target_weight)

            for output in outputs[1:]:
                pose_loss += pose_criterion(output, target_reg, target_weight)
                kd_pose_loss += kd_pose_criterion(output, toutput_reg, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target_reg, target_weight)
            kd_pose_loss = kd_pose_criterion(output, toutput_reg, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
        ##########################################################################################

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(pose_loss.item(), input.size(0))
        kd_pose_losses.update(kd_pose_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        ##########################################################################################
        if valid_type == 'hm':
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
        
        elif valid_type == 'reg':
            _, avg_acc, cnt, pred = accuracy_reg(output, target.detach().cpu().numpy(), beta)
        
        elif valid_type == 'reg_bias':
            _, avg_acc, cnt, pred = accuracy_reg_bias(output, target.detach().cpu().numpy(), beta)

        else:
            raise NotImplementedError
        ##########################################################################################
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def fpd_train_mid(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA
    #########################################
    valid_type = config.KD.VALID_TYPE
    beta = config.LOSS.BETA
    coarse_to_fine = config.LOSS.COARSE_TO_FINE
    #########################################

    # s_model switch to train mode and t_model switch to evaluate mode
    model.train()
    tmodel.eval()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)  # 网络输出是一个列表，因此对每一个stack的输出都进行loss运算，有多少个stack就算多少次
        # print("outputs", len(outputs))
        toutput = tmodel(input)
        # print("toutput", len(toutput))  # 一样与stack长度相等

        if isinstance(toutput, list):  # toutput只取最后一个stack的结果
            if coarse_to_fine:
                toutput = toutput[len(outputs):len(toutput)]
            else:
                toutput = toutput[-1]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        ##########################################################################################
        target_reg = meta['joints'][:,:,0:2]  # TODO 暂时不使用ground truth的坐标
        target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
        target_reg = target_reg / 256 - 0.5

        # toutput_reg, _ = get_max_preds(toutput.detach().cpu().numpy())  # 由于是分布蒸馏，因此蒸馏的不是坐标信息
        # toutput_reg = torch.Tensor(toutput_reg)
        # toutput_reg = toutput_reg.reshape((toutput_reg.shape[0], 16 * 2)).cuda()
        # toutput_reg = toutput_reg / 64 - 0.5

        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[0], target_reg, target_weight)
            if coarse_to_fine:
                kd_pose_loss = kd_pose_criterion(outputs[0], toutput[0], target_weight)
            else:
                kd_pose_loss = kd_pose_criterion(outputs[0], toutput, target_weight)

            for idx, output in enumerate(outputs[1:]):
                pose_loss += pose_criterion(output, target_reg, target_weight)
                if coarse_to_fine:
                    # print("idx", idx+1)
                    kd_pose_loss += kd_pose_criterion(output, toutput[idx+1], target_weight)
                else:
                    kd_pose_loss += kd_pose_criterion(output, toutput, target_weight)  # TODO 暂时不使用coarse-to-fine的策略
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target_reg, target_weight)
            kd_pose_loss = kd_pose_criterion(output, toutput, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
        ##########################################################################################

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(pose_loss.item(), input.size(0))
        kd_pose_losses.update(kd_pose_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        ##########################################################################################
        if valid_type == 'hm':
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
        
        elif valid_type == 'reg':
            # print("train_beta", beta)
            _, avg_acc, cnt, pred = accuracy_reg(output, target.detach().cpu().numpy(), beta)
        
        elif valid_type == 'reg_bias':
            _, avg_acc, cnt, pred = accuracy_reg_bias(output, target.detach().cpu().numpy(), beta)
        
        else:
            raise NotImplementedError
        ##########################################################################################
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def fpd_train_GThm_GTreg(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA
    #########################################
    valid_type = config.KD.VALID_TYPE
    beta = config.LOSS.BETA
    coarse_to_fine = config.LOSS.COARSE_TO_FINE
    #########################################

    # s_model switch to train mode and t_model switch to evaluate mode
    model.train()
    tmodel.eval()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)
        toutput = tmodel(input)

        if isinstance(toutput, list):
            if coarse_to_fine:
                toutput = toutput[len(outputs):len(toutput)]
            else:
                toutput = toutput[-1]

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        ##########################################################################################
        target_reg = meta['joints'][:,:,0:2]
        target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
        target_reg = target_reg / 256 - 0.5

        if isinstance(outputs, list):
            pose_loss = pose_criterion(outputs[0], target_reg, target_weight)
            kd_pose_loss = kd_pose_criterion(outputs[0], target, target_weight)

            for output in outputs[1:]:
                pose_loss += pose_criterion(output, target_reg, target_weight)
                kd_pose_loss += kd_pose_criterion(output, target, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
            output = outputs[-1]
        else:
            output = outputs
            pose_loss = pose_criterion(output, target_reg, target_weight)
            kd_pose_loss = kd_pose_criterion(output, target, target_weight)
            loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
        ##########################################################################################

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(pose_loss.item(), input.size(0))
        kd_pose_losses.update(kd_pose_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        ##########################################################################################
        if valid_type == 'hm':
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
        
        elif valid_type == 'reg':
            _, avg_acc, cnt, pred = accuracy_reg(output, target.detach().cpu().numpy(), beta)
        
        elif valid_type == 'reg_bias':
            _, avg_acc, cnt, pred = accuracy_reg_bias(output, target.detach().cpu().numpy(), beta)
        
        else:
            raise NotImplementedError
        ##########################################################################################
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


#################################################################################
#################################################################################
def train_dcj(config, train_loader, model, tmodel, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    valid_type = config.KD.VALID_TYPE
    train_type = config.KD.TRAIN_TYPE
    beta = config.LOSS.BETA
    coarse_to_fine = config.LOSS.COARSE_TO_FINE

    model.train()
    tmodel.eval()

    end = time.time()

    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        toutput = tmodel(input)
        if isinstance(toutput, list):
            if coarse_to_fine:
                toutput = toutput[len(outputs):len(toutput)]
            else:
                toutput = toutput[-1]

        toutput_reg, _ = get_max_preds(toutput.detach().cpu().numpy())
        toutput_reg = torch.Tensor(toutput_reg)
        toutput_reg = toutput_reg.reshape((toutput_reg.shape[0], 16 * 2)).cuda()
        toutput_reg = toutput_reg / 64 - 0.5

        target_reg = meta['joints'][:,:,0:2]
        target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
        target_reg = target_reg / 256 - 0.5

        if train_type == 'FPD_GTreg':
            if isinstance(outputs, list):
                loss = criterion(outputs[0], target_reg, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target_reg, target_weight)
                output = outputs[-1]
            else:
                output = outputs
                loss = criterion(output, target_reg, target_weight)
        
        elif train_type == 'FPD_reg':
            if isinstance(outputs, list):
                loss = criterion(outputs[0], toutput_reg, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, toutput_reg, target_weight)
                output = outputs[-1]
            else:
                output = outputs
                loss = criterion(output, toutput_reg, target_weight)

        elif train_type == 'FPD_GThm':
            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
                output = outputs[-1]
            else:
                output = outputs
                loss = criterion(output, target, target_weight)

        elif train_type == 'FPD_hm':
            if isinstance(outputs, list):
                loss = criterion(outputs[0], toutput, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, toutput, target_weight)
                output = outputs[-1]
            else:
                output = outputs
                loss = criterion(output, toutput, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def fpd_train_dcj(config, train_loader, model, tmodel, pose_criterion, kd_pose_criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pose_losses = AverageMeter()
    kd_pose_losses = AverageMeter()
    acc = AverageMeter()
    kd_weight_alpha = config.KD.ALPHA

    valid_type = config.KD.VALID_TYPE
    train_type = config.KD.TRAIN_TYPE
    beta = config.LOSS.BETA
    coarse_to_fine = config.LOSS.COARSE_TO_FINE

    model.train()
    tmodel.eval()

    end = time.time()

    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        outputs = model(input)
        
        if train_type in ('FPD_hm_GTreg', 'FPD_GThm_reg', 'FPD_hm_reg', 'FPD_GTreg_reg', 'FPD_GThm_hm'):  # 包括教师模型的
            toutput = tmodel(input)
            if isinstance(toutput, list):
                if coarse_to_fine:
                    toutput = toutput[len(outputs):len(toutput)]
                else:
                    toutput = toutput[-1]

        if train_type in ('FPD_GThm_reg', 'FPD_hm_reg', 'FPD_GTreg_reg'):  # 还包括教师模型reg的，子集关系
            toutput_reg, _ = get_max_preds(toutput.detach().cpu().numpy())
            toutput_reg = torch.Tensor(toutput_reg)
            toutput_reg = toutput_reg.reshape((toutput_reg.shape[0], 16 * 2)).cuda()
            toutput_reg = toutput_reg / 64 - 0.5

        if train_type in ('FPD_GThm_GTreg', 'FPD_GThm_reg',  'FPD_GThm_hm'):  # 包括GThm
            target = target.cuda(non_blocking=True)
            
        if train_type in ('FPD_GThm_GTreg', 'FPD_hm_GTreg', 'FPD_GTreg_reg'):  # 包括GTreg
            target_reg = meta['joints'][:,:,0:2]
            target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
            target_reg = target_reg / 256 - 0.5

        target_weight = target_weight.cuda(non_blocking=True)

        if train_type == 'FPD_GThm_GTreg':
            if isinstance(outputs, list):
                pose_loss = pose_criterion(outputs[0], target, target_weight)
                kd_pose_loss = kd_pose_criterion(outputs[0], target_reg, target_weight)

                for output in outputs[1:]:
                    pose_loss += pose_criterion(output, target, target_weight)
                    kd_pose_loss += kd_pose_criterion(output, target_reg, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
                output = outputs[-1]
            else:
                output = outputs
                pose_loss = pose_criterion(output, target, target_weight)
                kd_pose_loss = kd_pose_criterion(output, target_reg, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss

        elif train_type == 'FPD_hm_GTreg':
            if isinstance(outputs, list):
                kd_pose_loss = kd_pose_criterion(outputs[0], target_reg, target_weight)
                if coarse_to_fine:
                    pose_loss = pose_criterion(outputs[0], toutput[0], target_weight)
                else:
                    pose_loss = pose_criterion(outputs[0], toutput, target_weight)

                for idx, output in enumerate(outputs[1:]):
                    kd_pose_loss += kd_pose_criterion(output, target_reg, target_weight)
                    if coarse_to_fine:
                        pose_loss += pose_criterion(output, toutput[idx+1], target_weight)
                    else:
                        pose_loss += pose_criterion(output, toutput, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
                output = outputs[-1]
            else:
                output = outputs
                kd_pose_loss = kd_pose_criterion(output, target_reg, target_weight)
                pose_loss = pose_criterion(output, toutput, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
        
        elif train_type == 'FPD_GThm_reg':
            if isinstance(outputs, list):
                pose_loss = pose_criterion(outputs[0], target, target_weight)
                if coarse_to_fine:
                    kd_pose_loss = kd_pose_criterion(outputs[0], toutput_reg[0], target_weight)
                else:
                    kd_pose_loss = kd_pose_criterion(outputs[0], toutput_reg, target_weight)

                for idx, output in enumerate(outputs[1:]):
                    pose_loss += pose_criterion(output, target, target_weight)
                    if coarse_to_fine:
                        kd_pose_loss += kd_pose_criterion(output, toutput_reg[idx+1], target_weight)
                    else:
                        kd_pose_loss += kd_pose_criterion(output, toutput_reg, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
                output = outputs[-1]
            else:
                output = outputs
                pose_loss = pose_criterion(output, target, target_weight)
                kd_pose_loss = kd_pose_criterion(output, toutput_reg, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss

        elif train_type == 'FPD_hm_reg':
            if isinstance(outputs, list):
                if coarse_to_fine:
                    pose_loss = pose_criterion(outputs[0], toutput[0], target_weight)
                    kd_pose_loss = kd_pose_criterion(outputs[0], toutput_reg[0], target_weight)
                else:
                    pose_loss = pose_criterion(outputs[0], toutput, target_weight)
                    kd_pose_loss = kd_pose_criterion(outputs[0], toutput_reg, target_weight)

                for idx, output in enumerate(outputs[1:]):
                    if coarse_to_fine:
                        pose_loss += pose_criterion(output, toutput[idx+1], target_weight)
                        kd_pose_loss += kd_pose_criterion(output, toutput_reg[idx+1], target_weight)
                    else:
                        pose_loss += pose_criterion(output, toutput, target_weight)
                        kd_pose_loss += kd_pose_criterion(output, toutput_reg, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
                output = outputs[-1]
            else:
                output = outputs
                pose_loss = pose_criterion(output, toutput, target_weight)
                kd_pose_loss = kd_pose_criterion(output, toutput_reg, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
        
        elif train_type == 'FPD_GTreg_reg':
            if isinstance(outputs, list):
                pose_loss = pose_criterion(outputs[0], target_reg, target_weight)
                if coarse_to_fine:
                    kd_pose_loss = kd_pose_criterion(outputs[0], toutput_reg[0], target_weight)
                else:
                    kd_pose_loss = kd_pose_criterion(outputs[0], toutput_reg, target_weight)

                for idx, output in enumerate(outputs[1:]):
                    pose_loss += pose_criterion(output, target_reg, target_weight)
                    if coarse_to_fine:
                        kd_pose_loss += kd_pose_criterion(output, toutput_reg[idx+1], target_weight)
                    else:
                        kd_pose_loss += kd_pose_criterion(output, toutput_reg, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
                output = outputs[-1]
            else:
                output = outputs
                pose_loss = pose_criterion(output, target_reg, target_weight)
                kd_pose_loss = kd_pose_criterion(output, toutput_reg, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss

        elif train_type == 'FPD_GThm_hm':
            if isinstance(outputs, list):
                pose_loss = pose_criterion(outputs[0], target, target_weight)
                if coarse_to_fine:
                    kd_pose_loss = kd_pose_criterion(outputs[0], toutput[0], target_weight)
                else:
                    kd_pose_loss = kd_pose_criterion(outputs[0], toutput, target_weight)

                for idx, output in enumerate(outputs[1:]):
                    pose_loss += pose_criterion(output, target, target_weight)
                    if coarse_to_fine:
                        kd_pose_loss += kd_pose_criterion(output, toutput[idx+1], target_weight)
                    else:
                        kd_pose_loss += kd_pose_criterion(output, toutput, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss
                output = outputs[-1]
            else:
                output = outputs
                pose_loss = pose_criterion(output, target, target_weight)
                kd_pose_loss = kd_pose_criterion(output, toutput, target_weight)
                loss = (1 - kd_weight_alpha) * pose_loss + kd_weight_alpha * kd_pose_loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        pose_losses.update(pose_loss.item(), input.size(0))
        kd_pose_losses.update(kd_pose_loss.item(), input.size(0))
        losses.update(loss.item(), input.size(0))

        if valid_type == 'hm':
            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
        
        elif valid_type == 'reg':
            # print("train_beta", beta)
            _, avg_acc, cnt, pred = accuracy_reg(output, target.detach().cpu().numpy(), beta)
        
        elif valid_type == 'reg_bias':
            _, avg_acc, cnt, pred = accuracy_reg_bias(output, target.detach().cpu().numpy(), beta)
        
        else:
            raise NotImplementedError

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'POSE_Loss {pose_loss.val:.5f} ({pose_loss.avg:.5f})\t' \
                  'KD_POSE_Loss {kd_pose_loss.val:.5f} ({kd_pose_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,data_time=data_time,
                      pose_loss=pose_losses, kd_pose_loss=kd_pose_losses, loss=losses,
                      acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_pose_loss', pose_losses.val, global_steps)
            writer.add_scalar('train_kd_pose_loss', kd_pose_losses.val, global_steps)
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
#################################################################################
#################################################################################


def validate_reg(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    #########################################
    valid_type = config.KD.VALID_TYPE
    beta = config.LOSS.BETA
    #########################################

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            ##########################################################################################
            if valid_type == 'hm':
                loss = criterion(output, target, target_weight)
                num_images = input.size(0)
                losses.update(loss.item(), num_images)

                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                 target.cpu().numpy())

            elif valid_type == 'reg':
                target_reg = meta['joints'][:,:,0:2]
                target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
                target_reg = target_reg / 256 - 0.5

                loss = criterion(output, target_reg, target_weight)
                num_images = input.size(0)
                losses.update(loss.item(), num_images)

                # print("valid beta", beta)
                _, avg_acc, cnt, pred = accuracy_reg(output, target.cpu().numpy(), beta)

            elif valid_type == 'reg_bias':
                target_reg = meta['joints'][:,:,0:2]
                target_reg = target_reg.reshape((target_reg.shape[0], 16 * 2)).cuda()
                target_reg = target_reg / 256 - 0.5

                loss = criterion(output, target_reg, target_weight)
                num_images = input.size(0)
                losses.update(loss.item(), num_images)

                # print("valid beta", beta)
                _, avg_acc, cnt, pred = accuracy_reg_bias(output, target.cpu().numpy(), beta)

            else:
                raise NotImplementedError
            ##########################################################################################

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            ##########################################################
            if valid_type == 'hm':
                preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)
            
            elif valid_type == 'reg':
                preds, maxvals = get_final_preds_reg(
                    config, output.clone().cpu().numpy(), c, s, beta)
            
            elif valid_type == 'reg_bias':
                preds, maxvals = get_final_preds_reg_bias(
                    config, output.clone().cpu().numpy(), c, s, beta)

            else:
                raise NotImplementedError
            ##########################################################

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
