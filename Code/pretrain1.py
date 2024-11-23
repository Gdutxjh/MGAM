import argparse
import os
import numpy as np
import random
import time
import datetime
import json
import ruamel.yaml as yaml
from pathlib import Path
from tqdm import tqdm
import pickle as pkl

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from data import create_dataset, create_loader, create_CPT_dataset
from data.PathVQA_dataset import vqa_collate_fn
from models.bivi_vqa import BiVision_VQA
from models.vqa3 import BiVision_VQA3, BiVision_VQA6, BiVision_VQA7, BiVision_VQA8, BiVision_VQA9, BiVision_VQA10
from models.vqa13 import Bistep_vqa_pretrain, Bistep_vqa
from data.image_enhancement import image_enhance
from utils import cosine_lr_schedule, calculate_bleu_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_similarity(batch_anchor, label_feat):
    # cos distance
    similarity = batch_anchor.mm(label_feat.t())   # b * v   # 64 * 5479
    similarity = torch.softmax(similarity, dim=1)
    return similarity

def loss_function(x_hat, x, mu, log_var):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)

    # 3. total loss
    loss = BCE + KLD
    return loss, BCE, KLD

def get_keywords(args):
    with open(os.path.join('bert-base-uncased', 'bert-base-uncased-vocab.txt'), 'r') as f:
        key = f.read()

    keywords = key.split()

    return keywords

def get_acc(preds, gts, qids, idx2ans):
    total_acc = (preds == gts).mean() * 100.
    # total_bleu = calculate_bleu_score(preds, gts, idx2ans)
    acc = {'val_acc': np.round(total_acc, 4)}
    close_ans = ['yes', 'no', 'Yes', 'No']
    # bleu = {'total_bleu': np.round(total_bleu, 4)}
    
    result = []
    
    return result, acc

def train4(config, model, data_loader, optimizer, loss_fn, epoch, device, train_size, id2ans, MSE_loss=None):
    model.train()
    train_loss = []
    close_ans = ['yes', 'no', 'Yes', 'No']
    preds = []
    gts = []
    # for i, (qid, image, tokens, segment_ids, input_mask, answer, answer_type) in enumerate(data_loader):
    for i, (qid, image, tokens, segment_ids, input_mask, text) in enumerate(data_loader):
        # qid: list  image: torch[8, 3, 480, 480] tokens: torch[8,20] 
        # segment_ids: torch[8,20]  input_mask: torch[8,20] answer: torch[8]  weights: torch[8] 
        
        # image = image_enhance(image)
        alpha = config['alpha']*min(1,(epoch*len(data_loader)+i)/(2*len(data_loader))) 
        image = image.to(device) # bs*3*480*480 
        tokens, segment_ids, input_mask = tokens.to(device), segment_ids.to(device), input_mask.to(device)
        logits, loss, labels = model(image, tokens, segment_ids, input_mask, text=text, alpha=alpha)

        loss_np = loss.detach().cpu().numpy()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        print("\r[epoch: %d][%d/%d] loss: %f" % (epoch, i, train_size/config["batch_size_train"], loss_np), end="        ")

        bool_label = labels > 0
        pred = logits[bool_label, :].argmax(1)
        valid_labels = labels[bool_label] 
      
        preds.append(pred)
        gts.append(valid_labels)
        train_loss.append(float(loss_np))
    
    preds = torch.cat(preds).cpu().numpy()
    gts = torch.cat(gts).cpu().numpy()
    train_loss = np.mean(train_loss)

    total_acc = (preds == gts).mean() * 100.
    acc = {'total_acc': np.round(total_acc, 4)}

    return acc, train_loss

def main(args, config):
    device = args.device

    # seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    rand_a = int(time.strftime("%Y%m%d%H%M%S")) % 114514

    # ans2id = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))[0]
    # id2ans = json.load(open(os.path.join(config['vqa_root'], 'id2ans.json')))
    # #
    # ans_embeds_all = []
    # ans_embeds = json.load(open(os.path.join(config['vqa_root'], config['ans_embeds'])))[0]
    # for ans, embeds in ans_embeds.items():
    #     ans_embeds_all.append(embeds)
    # ans_embeds_all = torch.tensor(ans_embeds_all)
    
    # id2ans = json.load(open(os.path.join(config['vqa_root'], 'en_data/ids2ans.json')))
    id2ans = json.load(open(os.path.join(config['vqa_root'], config['label2ans'])))
    # ans_size = len(id2ans)

    # loading dataset
    print("Creating medical vqa datasets")
    datasets = create_dataset(config=config)
    # ans_size = len(id2ans)
    # keywords = json.load(open(os.path.join(config['vqa_root'], config['keywords']), "r"))
    keywords = get_keywords(args)
    ans_size = len(keywords)
    train_size, val_size, test_size = datasets[0].ann_size, datasets[1].ann_size, datasets[2].ann_size

    train_loader, val_loader, test_loader = create_loader(datasets,[None, None, None],
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4,4,4], is_trains=[True, False, False]) 

    # Model
    print("Creating model")
    if args.model == 'vqa3':
        model = BiVision_VQA3(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    elif args.model == 'vqa6':
        model = BiVision_VQA6(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    elif args.model == 'vqa7':
        model = BiVision_VQA7(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    elif args.model == 'vqa8':
        model = BiVision_VQA8(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    elif args.model == 'vqa9':
        model = BiVision_VQA9(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    elif args.model == 'vqa10':
        model = BiVision_VQA10(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    elif args.model == 'vqa16':
        model = Bistep_vqa_pretrain(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    else:
        model = BiVision_VQA(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)

    if args.load_checkpoint:
        pretrained_dict = torch.load(args.load_checkpoint, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)} #  and 'fusion_layer' not in k and "classifier1.2" not in k and "classifier2.2" not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load(args.load_checkpoint), strict=False)
    
    model.to(args.device)

    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    loss_fn = nn.CrossEntropyLoss()
    MSE_loss = nn.MSELoss()

    best_acc = 0
    best_epoch = 0
    print("Start training")
    start_time = 0

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            acc, train_loss = train4(config, model, train_loader, optimizer, loss_fn, epoch, device, train_size, id2ans)
            print("epoch: %d," % epoch, end=" ")
            print(acc, end="\n    ")
            
            ###############3
            if acc['total_acc'] > best_acc:
                model_path = os.path.join(args.save_dir, args.dataset+"_"+str(seed)+"_best.pt")
                torch.save(model.state_dict(), model_path)
                print("save model to %s" % model_path)
                best_acc = acc['total_acc']
            ###########33
        else:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/VQA_RAD.yaml', type=str)
    parser.add_argument('--output_dir', default='output/VQA_RAD', type=str)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=29, type=int)
    parser.add_argument('--load_checkpoint', 
                        # default="ckpt/PathVQA_45_best.pt",
                        # default="ckpt/VQA_RAD_34_best.pt",
                        # default="ckpt/pretrain/PathVQA_29_best.pt",
                        default="ckpt/pretrain/VQA_RAD_29_best.pt",
                        # 35question  34mask
                        type=str)
    parser.add_argument('--save_dir', default="ckpt/pretrain", type=str)
    parser.add_argument('--dataset', default="VQA_RAD", type=str)
    parser.add_argument('--model', default='vqa16', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)

