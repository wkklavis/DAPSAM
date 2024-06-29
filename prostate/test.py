import os
import sys

import cv2
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from segment_anything.utils.metrics import calculate_metrics
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry


def inference(args, epoch, snapshot_path, test_loader, model, test_save_path=None):
    print("\nTesting and Saving the results...")
    print("--" * 15)
    metrics_y = [[], []]
    metric_dict = ['Dice', 'ASD']

    last_name = None

    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader, position=0, leave=True, ncols=70)):
            x, y, path = data['data'], data['mask'], data['name']

            current_name = path
            if last_name is None:
                last_name = path

            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)

            x = x.cuda()
            seg_logit = model(x, False, args.img_size)
            seg_output = torch.sigmoid(seg_logit['masks'].detach().cpu())
            
            if current_name != last_name:  # Calculate the previous 3D volume
                metrics = calculate_metrics(seg_output3D, y3D)
                for i in range(len(metrics)):
                    metrics_y[i].append(metrics[i])

                del seg_output3D
                del y3D

            try:
                seg_output3D = torch.cat((seg_output.unsqueeze(2), seg_output3D), 2)
                y3D = torch.cat((y.unsqueeze(2), y3D), 2)
            except:
                seg_output3D = seg_output.unsqueeze(2)
                y3D = y.unsqueeze(2)

            last_name = current_name

    # Calculate the last 3D volume
    metrics = calculate_metrics(seg_output3D, y3D)
    for i in range(len(metrics)):
        metrics_y[i].append(metrics[i])

    test_metrics_y = np.mean(metrics_y, axis=1)
    print_test_metric = {}
    for i in range(len(test_metrics_y)):
        print_test_metric[metric_dict[i]] = test_metrics_y[i]

    with open(snapshot_path + '/' + 'test_' + args.Source_Dataset + '_to'+ '.txt', 'a', encoding='utf-8') as f:
        f.write('Epoch '+str(epoch)+' Test Metrics:\n')
        f.write(str(print_test_metric) + '\n')  # Dice

    logging.info("Test Metrics: "+str(print_test_metric))
    return test_metrics_y


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


