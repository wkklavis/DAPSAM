import os
import sys

import cv2
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from datasets.fundus.RIGA_dataloader import RIGA_labeled_set
from datasets.fundus.convert_csv_to_list import convert_labeled_list
from datasets.fundus.dice import get_hard_dice
from datasets.fundus.transform import collate_fn_ts
from segment_anything.utils.metrics import calculate_metrics
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry


def inference_riga(args, epoch, snapshot_path, test_loader, model, test_save_path=None):
    print("\nTesting and Saving the results...")
    print("--" * 15)
    val_disc_dice_list = list()
    val_cup_dice_list = list()
    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader, position=0, leave=True, ncols=70)):
            x, y = data['data'], data['seg']

            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)

            x = x.cuda()
            seg_logit = model(x, False, args.img_size)
            seg_output = torch.sigmoid(seg_logit['masks'].detach().cpu())

            val_disc_dice_list.append(get_hard_dice(seg_output[:, 0].cpu(), (y[:, 0] > 0).cpu() * 1.0))
            val_cup_dice_list.append(get_hard_dice(seg_output[:, 1].cpu(), (y[:, 0] == 2).cpu() * 1.0))

    mean_val_disc_dice = np.mean(val_disc_dice_list)
    mean_val_cup_dice = np.mean(val_cup_dice_list)

    logging.info('Epoch{}  Val disc dice: {}; Cup dice: {}'.format(epoch, mean_val_disc_dice, mean_val_cup_dice))

    with open(snapshot_path + '/' + 'test_' + args.Source_Dataset + '_to'+ '.txt', 'a', encoding='utf-8') as f:
        f.write('Epoch '+str(epoch)+' Test Metrics:\n')
        f.write(str('Val disc dice: {}; Cup dice: {}'.format(mean_val_disc_dice, mean_val_cup_dice)) + '\n')  # Dice

    return mean_val_disc_dice, mean_val_cup_dice

def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--root_path', type=str,
                        default='./data/datasets/RIGAPlus', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='RIGA', help='experiment_name')
    parser.add_argument('--Source_Dataset', type=str, default='BinRushed',
                        help='BinRushed/Magrabia')
    parser.add_argument('--Target_Dataset', nargs='+', type=str,
                        default=['MESSIDOR_Base1', 'MESSIDOR_Base2', 'MESSIDOR_Base3'],
                        help='MESSIDOR_Base1/MESSIDOR_Base2/MESSIDOR_Base3')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--output', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')

    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='./pretrained/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')

    parser.add_argument('--snapshot', type=str, default='./snapshot/DAPSAM-BinRushed.pth', help='model snapshot')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()


    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'FUNDUS': {
            'Dataset': args.dataset,
            'num_classes': args.num_classes,
        }
    }
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    net = sam.cuda()
    if args.snapshot is not None:
        weights = torch.load(args.snapshot)
        net.load_state_dict(weights)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))



    #to rest target domain
    target_name = args.Target_Dataset
    valid_loader = []
    for t_n in target_name:
        target_csv = [os.path.join(args.root_path, t_n + '.csv')]
        ts_img_list, ts_label_list = convert_labeled_list(target_csv, r=1)

        ts_dataset = RIGA_labeled_set(args.root_path, ts_img_list, ts_label_list)
        valid_loader.append(torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=0,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn_ts))

    mean_val_disc_dice, mean_val_cup_dice = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
                                                           test_loader=valid_loader[0], model=net,
                                                           test_save_path=None)

    mean_val_disc_dice1, mean_val_cup_dice1 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
                                                             test_loader=valid_loader[1], model=net,
                                                             test_save_path=None)

    mean_val_disc_dice2, mean_val_cup_dice2 = inference_riga(args=args, epoch=0, snapshot_path=log_folder,
                                                             test_loader=valid_loader[2], model=net,
                                                             test_save_path=None)
