import argparse
import os
import numpy as np
import random
import time
import json
import ruamel.yaml as yaml
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F


from data import create_dataset, create_loader
from models.vqa20 import VQA10
from utils import cosine_lr_schedule
from swin_unet.config import get_config


def train(config, model, data_loader, optimizer, loss_fn, epoch, device, train_size, id2ans):
    model.train()
    train_loss = []
    close_ans = ['yes', 'no', 'Yes', 'No']
    preds = []
    gts = []
    preds_top = []
    for i, (qid, img, image, tokens, segment_ids, input_mask, answer, question) in enumerate(data_loader):
        img, image, answer = img.to(device), image.to(device), answer.to(device) # bs*3*480*480 
        tokens, segment_ids, input_mask = tokens.to(device), segment_ids.to(device), input_mask.to(device)
        logits, loss = model(image, tokens, segment_ids, input_mask, answer, id2ans, question=question)
    

def main(args, config, swin_config):
    device = args.device
    torch.backends.cudnn.enabled = False
    config["pretrain"] = False
    # seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # dataset
    print("Creating medical vqa datasets")
    id2ans = json.load(open(os.path.join(config['vqa_root'], config['label2ans'])))
    ans_size = len(id2ans)
    datasets = create_dataset(config=config)
    train_size, val_size, test_size = datasets[0].ann_size, datasets[1].ann_size, datasets[2].ann_size
    train_loader, val_loader, test_loader = create_loader(datasets,[None, None, None],
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4,4,4], is_trains=[True, False, False]) 
    
    # model
    print("Creating model")
    if args.model == 'vqa20':
        model = VQA10(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device, config=config, swin_cofig=swin_config)
    else:
        model = VQA10(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device, config=config, swin_cofig=swin_config)
        
    if args.load_checkpoint:
        pretrained_dict = torch.load(args.load_checkpoint, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)} #  and k.split(".")[0] not in layers_out)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
    model.to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    loss_fn = nn.CrossEntropyLoss()
    
    best_acc = 0
    print("Start training")
    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            acc, train_loss = train()
    

def init_swin_config(swin_config, config):
    dataset_config = {
        'Synapse': {
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    
    swin_config.DATA.IMG_SIZE = config["image_size"]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/VQA_RAD.yaml', type=str)
    parser.add_argument('--output_dir', default='output/VQA_RAD', type=str)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--seed', default=53, type=int)
    parser.add_argument('--load_checkpoint', default=None,
                        type=str)
    parser.add_argument('--save_dir', default="ckpt", type=str)
    parser.add_argument('--dataset', default="VQA_RAD", type=str)
    parser.add_argument('--model', default='vqa20', type=str) 
    
    ##########################
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, default="./swin_unet/configs/swin_tiny_patch4_window7_224_lite.yaml", 
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    swin_config = yaml.load(open(args.swin_config, 'r'), Loader=yaml.Loader)
    swin_config = get_config(args)
    args.result_dir = os.path.join(args.output_dir, 'result')
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True) 
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config, swin_config)
    
    