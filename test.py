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
    # test1_prototype = []
    # test1_di_prototype = []
    # test2_prototype = []
    # test2_di_prototype = []
    # test3_prototype = []
    # test3_di_prototype = []
    # test4_prototype = []
    # test4_di_prototype = []
    # test5_prototype = []
    # test5_di_prototype = []
    # source_prototype = []
    # source_di_prototype = []
    # test1_weight = []
    # test2_weight = []
    # test3_weight = []
    # test4_weight = []
    # test5_weight = []
    # source_weight = []
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
            ####################
            # from thop import profile
            # from fvcore.nn import FlopCountAnalysis, parameter_count_table
            # # x = torch.randn(1, 256, 24, 24).float().cuda()
            #
            # trainable_params = 0
            # all_param = 0
            # for _, param in model.named_parameters():
            #     all_param += param.numel()
            #     if param.requires_grad:
            #         trainable_params += param.numel()
            # print(
            #     f"trainable params: {trainable_params/1e6} || all params: {all_param/1e6}")
            #
            #
            # macs, params = profile(model, inputs=(x,False, args.img_size))  # ,verbose=False
            # print("MACs", macs)
            # print("p", params)
            #
            # print("@@@@@@@@@@@@@@")
            #
            # # flops = FlopCountAnalysis(model, x)
            # # print("FLOPs", flops.total())
            # print(parameter_count_table(model))
            ###################
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

            # if ("RUNMC") in current_name[0] and len(source_di_prototype)<100:
            #     source_prototype.append(model.prompt_generator.prototype.cpu().detach().numpy())
            #     source_di_prototype.append(model.prompt_generator.di_prototype.cpu().detach().numpy())
            #
            # if ("BMC") in current_name[0] and len(test1_di_prototype)<100:
            #     test1_prototype.append(model.prompt_generator.prototype.cpu().detach().numpy())
            #     test1_di_prototype.append(model.prompt_generator.di_prototype.cpu().detach().numpy())
            #
            # if ("BIDMC") in current_name[0] and len(test2_di_prototype)<100:
            #     test2_prototype.append(model.prompt_generator.prototype.cpu().detach().numpy())
            #     test2_di_prototype.append(model.prompt_generator.di_prototype.cpu().detach().numpy())
            #
            # if ("UCL") in current_name[0] and len(test3_di_prototype)<100:
            #     test3_prototype.append(model.prompt_generator.prototype.cpu().detach().numpy())
            #     test3_di_prototype.append(model.prompt_generator.di_prototype.cpu().detach().numpy())
            #
            # if ("HK") in current_name[0] and len(test4_di_prototype)<100:
            #     test4_prototype.append(model.prompt_generator.prototype.cpu().detach().numpy())
            #     test4_di_prototype.append(model.prompt_generator.di_prototype.cpu().detach().numpy())
            #
            # if ("I2CVB") in current_name[0] and len(test5_di_prototype)<100:
            #     test5_prototype.append(model.prompt_generator.prototype.cpu().detach().numpy())
            #     test5_di_prototype.append(model.prompt_generator.di_prototype.cpu().detach().numpy())
            #
            # if len(test4_di_prototype)== 100 and len(test5_di_prototype)==100:
            #     test1_prototype = np.array(test1_prototype)
            #     test1_di_prototype = np.array(test1_di_prototype)
            #
            #     test2_prototype = np.array(test2_prototype)
            #     test2_di_prototype = np.array(test2_di_prototype)
            #
            #     test3_prototype = np.array(test3_prototype)
            #     test3_di_prototype = np.array(test3_di_prototype)
            #
            #     test4_prototype = np.array(test4_prototype)
            #     test4_di_prototype = np.array(test4_di_prototype)
            #
            #     test5_prototype = np.array(test5_prototype)
            #     test5_di_prototype = np.array(test5_di_prototype)
            #
            #     source_prototype = np.array(source_prototype)
            #     source_di_prototype = np.array(source_di_prototype)
            #
            #     memory_bank = model.prompt_generator.memory_bank.weight.cpu().detach().numpy()
            #
            #     # all_data = np.vstack([test1_prototype, test2_prototype,
            #     #                       test3_prototype, test4_prototype,
            #     #                       test5_prototype, source_prototype ])
            #     # all_data = np.vstack([test1_di_prototype, test2_di_prototype,
            #     #                       test3_di_prototype, test4_di_prototype,
            #     #                       test5_di_prototype, source_di_prototype, memory_bank])
            #     all_data = np.vstack([test1_prototype, test1_di_prototype, test2_prototype, test2_di_prototype,
            #                           test3_prototype, test3_di_prototype, test4_prototype, test4_di_prototype,
            #                           test5_prototype, test5_di_prototype,  memory_bank])#source_prototype, source_di_prototype,
            #
            #     # 使用 t-SNE 进行降维
            #     tsne = TSNE(n_components=2, random_state=42, perplexity=20)
            #     tsne_result = tsne.fit_transform(all_data)
            #     # 分离每个类别的 t-SNE 结果
            #     tsne_class1 = tsne_result[:100]  # 假设每个类别有 100 个数据点
            #     tsne_class2 = tsne_result[100:200]
            #     tsne_class3 = tsne_result[200:300]
            #     tsne_class4 = tsne_result[300:400]
            #     tsne_class5 = tsne_result[400:500]
            #     tsne_class6 = tsne_result[500:600]
            #     tsne_class7 = tsne_result[600:700]
            #     tsne_class8 = tsne_result[700:800]
            #     tsne_class9 = tsne_result[800:900]
            #     tsne_class10 = tsne_result[900:1000]
            #
            #     # tsne_class11 = tsne_result[1000:1100]
            #     # tsne_class12 = tsne_result[1100:1200]
            #
            #     tsne_class11 = tsne_result[1000:1200]
            #     # 绘制 t-SNE 图
            #     plt.figure(figsize=(15, 10))
            #     plt.scatter(tsne_class1[:, 0], tsne_class1[:, 1], label='BMC_Prototype', color='pink', alpha=0.7,)
            #     plt.scatter(tsne_class2[:, 0], tsne_class2[:, 1], label='BMC_DA_Prototype', color='pink', marker = "^", alpha=0.7,)
            #
            #     plt.scatter(tsne_class9[:, 0], tsne_class9[:, 1], label='I2CVB_Prototype', color='purple',alpha=0.7)
            #     plt.scatter(tsne_class10[:, 0], tsne_class10[:, 1], label='I2CVB_DA_Prototype', color='purple', marker = "^",alpha=0.7,)
            #
            #
            #     plt.scatter(tsne_class5[:, 0], tsne_class5[:, 1], label='UCL_Prototype', color='green',alpha=0.7,)
            #     plt.scatter(tsne_class6[:, 0], tsne_class6[:, 1], label='UCL_DA_Prototype', color='green', marker = "^", alpha=0.7,)
            #
            #     plt.scatter(tsne_class3[:, 0], tsne_class3[:, 1], label='BIDMC_Prototype', color='orange', alpha=0.7,)
            #     plt.scatter(tsne_class4[:, 0], tsne_class4[:, 1], label='BIDMC_DA_Prototype', color='orange', marker = "^", alpha=0.7,)
            #
            #     plt.scatter(tsne_class7[:, 0], tsne_class7[:, 1], label='HK_Prototype', color='brown')
            #     plt.scatter(tsne_class8[:, 0], tsne_class8[:, 1], label='HK_DA_Prototype', color='brown', marker = "^" ,alpha=0.7,)
            #
            #     # plt.scatter(tsne_class11[:, 0], tsne_class11[:, 1], label='Source_Prototype', color='olive')
            #     # plt.scatter(tsne_class12[:, 0], tsne_class12[:, 1], label='Source_DA_Prototype', color='olive', marker = "^" ,alpha=0.7,)
            #
            #     plt.scatter(tsne_class11[:, 0], tsne_class11[:, 1], label='Memory_Bank_Prototype', color='gray',alpha=0.7,)
            #
            #     plt.title('tSNE Prototype Visualization', fontdict={'size': 18})
            #     plt.legend()
            #     plt.grid(True)
            #     plt.savefig('./result/PrototypeVisualization20-1.png')
            #     plt.show()
            # if ("RUNMC") in current_name[0] and len(source_weight)<100:
            #     source_weight.append(model.prompt_generator.memory_bank.att_weight.cpu().detach().numpy())
            #
            # if ("BMC") in current_name[0] and len(test1_weight)<100:
            #     test1_weight.append(model.prompt_generator.memory_bank.att_weight.cpu().detach().numpy())
            #
            # if ("BIDMC") in current_name[0] and len(test2_weight)<100:
            #     test2_weight.append(model.prompt_generator.memory_bank.att_weight.cpu().detach().numpy())
            #
            # if ("UCL") in current_name[0] and len(test3_weight)<100:
            #     test3_weight.append(model.prompt_generator.memory_bank.att_weight.cpu().detach().numpy())
            #
            # if ("HK") in current_name[0] and len(test4_weight)<100:
            #     test4_weight.append(model.prompt_generator.memory_bank.att_weight.cpu().detach().numpy())
            #
            # if ("I2CVB") in current_name[0] and len(test5_weight)<100:
            #     test5_weight.append(model.prompt_generator.memory_bank.att_weight.cpu().detach().numpy())
            #
            # if len(test4_weight)== 100 and len(test5_weight)==100:
            #     test1_weight = np.array(test1_weight)
            #
            #     test2_weight = np.array(test2_weight)
            #
            #     test3_weight = np.array(test3_weight)
            #
            #     test4_weight = np.array(test4_weight)
            #
            #     test5_weight = np.array(test5_weight)
            #
            #     source_weight = np.array(source_weight)
            #
            #     all_data = np.vstack([test1_weight, test2_weight,test3_weight,test4_weight,test5_weight,source_weight])
            #     # 使用 t-SNE 进行降维
            #     tsne = TSNE(n_components=2, random_state=42)
            #     tsne_result = tsne.fit_transform(all_data)
            #     # 分离每个类别的 t-SNE 结果
            #     tsne_class1 = tsne_result[:100]  # 假设每个类别有 100 个数据点
            #     tsne_class2 = tsne_result[100:200]
            #     tsne_class3 = tsne_result[200:300]
            #     tsne_class4 = tsne_result[300:400]
            #     tsne_class5 = tsne_result[400:500]
            #     tsne_class6 = tsne_result[500:600]
            #
            #
            #     # 绘制 t-SNE 图
            #     plt.figure(figsize=(15, 10))
            #     plt.scatter(tsne_class1[:, 0], tsne_class1[:, 1], label='test1_weight', color='red')
            #     plt.scatter(tsne_class2[:, 0], tsne_class2[:, 1], label='test2_weight', color='blue')
            #     plt.scatter(tsne_class3[:, 0], tsne_class3[:, 1], label='test3_weight', color='green')
            #     plt.scatter(tsne_class4[:, 0], tsne_class4[:, 1], label='test4_weight', color='yellow')
            #
            #     plt.scatter(tsne_class5[:, 0], tsne_class5[:, 1], label='test5_weight', color='purple')
            #     plt.scatter(tsne_class6[:, 0], tsne_class6[:, 1], label='source_weight', color='black')
            #
            #     plt.title('t-SNE Visualization of 6 weights')
            #     plt.legend()
            #     plt.grid(True)
            #     plt.savefig('./result/weight_tsne.png')
            #     plt.show()
            draw_output = (seg_output.detach().cpu().numpy() * 255).astype(np.uint8)
            # cv2.imwrite(args.result_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_pred.png',
            #             draw_output[0][0])
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

def save_image(args, epoch, snapshot_path, test_loader, model, test_save_path=None):
    print("\nSave images")
    print("--" * 15)
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
                del seg_output3D
                del y3D

            try:
                seg_output3D = torch.cat((seg_output.unsqueeze(2), seg_output3D), 2)
                y3D = torch.cat((y.unsqueeze(2), y3D), 2)
            except:
                seg_output3D = seg_output.unsqueeze(2)
                y3D = y.unsqueeze(2)

            output_path = os.path.join(snapshot_path, 'saved_imges',str(path[0]).split('/')[-2])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            draw_output = (seg_output.detach().cpu().numpy() * 255).astype(np.uint8)
            draw_x = (x.detach().cpu().numpy() * 255).astype(np.uint8)
            draw_gt = (y.detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(output_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_pred.png',
                        draw_output[0][0])
            cv2.imwrite(output_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_x.png',
                        draw_x[0][0])
            cv2.imwrite(output_path + '/' + str(path[0]).split('/')[-1] + '-' + str(y3D.shape[2]) + '_gt.png',
                        draw_gt[0][0])
            last_name = current_name
    logging.info("saved")
    return

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
    parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
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
        'PROSTATE': {
            'Dataset': args.dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)
