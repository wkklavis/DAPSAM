import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from datasets.prostate.PROSTATE_dataloader import PROSTATE_dataset
from datasets.prostate.convert_csv_to_list import convert_labeled_list
from datasets.prostate.normalize import normalize_image
from datasets.prostate.transform import collate_fn_w_transform, collate_fn_wo_transform
from test import inference
from utils import DiceLoss, Focal_loss, dice_coeff, bce_loss
from torchvision import transforms



def calc_loss(outputs, label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    logits = outputs['masks']
    pred = torch.nn.Sigmoid()(logits)
    loss_ce = bce_loss(pred=pred, label=label_batch)
    loss_dice = dice_coeff(pred=pred, label=label_batch)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def trainer_prostate(args, model, snapshot_path, multimask_output, low_res):
    #single-source-domain
    source_csv = [args.Source_Dataset + '.csv']
    sr_img_list, sr_label_list = convert_labeled_list(args.root_path, source_csv)

    print('Training Phase')
    print(source_csv)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    source_dataset = PROSTATE_dataset(args.root_path, sr_img_list, sr_label_list,
                                      args.img_size, args.batch_size, img_normalize=False)
    trainloader = DataLoader(dataset=source_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   collate_fn=collate_fn_w_transform,
                                   num_workers=args.num_workers)


    #to rest target domain
    target_name = args.Target_Dataset
    target_csv = []
    for t_n in target_name:
        target_csv.append(t_n + '.csv')
    ts_img_list, ts_label_list = convert_labeled_list(args.root_path, target_csv)

    target_valid_dataset = PROSTATE_dataset(args.root_path, ts_img_list, ts_label_list,
                                            args.img_size, img_normalize=True)
    valid_loader = DataLoader(dataset=target_valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              collate_fn=collate_fn_wo_transform,
                              num_workers=args.num_workers)


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = bce_loss
    dice_loss = dice_coeff
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = (range(max_epoch))
    for epoch_num in iterator:
        logging.info('Epoch %d / %d' % (epoch_num, max_epoch))
        for batch, data in enumerate(trainloader):
            x, y = data['data'], data['mask']
            x = torch.from_numpy(normalize_image(x)).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)

            x, y = x.cuda(), y.cuda()

            outputs = model(x, multimask_output, args.img_size)

            loss, loss_ce, loss_dice = calc_loss(outputs, y, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            #test
            result_list = inference(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=valid_loader, model=model, test_save_path=None)
            writer.add_scalar('Valid_Dice', result_list[0], (epoch_num + 1) // save_interval)
            writer.add_scalar('Valid_ASD', result_list[1], (epoch_num + 1) // save_interval)

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:

            # save model
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            break

    writer.close()
    return "Training Finished!"
