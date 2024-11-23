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
from models.vqa3 import BiVision_VQA3, BiVision_VQA6, BiVision_VQA7, BiVision_VQA8, BiVision_VQA9
from data.image_enhancement import image_enhance
from utils import cosine_lr_schedule, calculate_bleu_score

def get_acc(preds, gts, qids, idx2ans):
    total_acc = (preds == gts).mean() * 100.
    # total_bleu = calculate_bleu_score(preds, gts, idx2ans)
    acc = {'val_acc': np.round(total_acc, 4)}
    close_ans = ['yes', 'no', 'Yes', 'No']
    # bleu = {'total_bleu': np.round(total_bleu, 4)}
    
    result = []
    yn_acc = 0
    other_acc = 0
    yn_total = 0.001
    other_total = 0.001
    for i in range(len(preds)):
        pred = idx2ans[int(preds[i])]
        gt = idx2ans[int(gts[i])]
        result.append({
            "qid": int(qids[i]),
            "pred_id": int(preds[i]),
            "gt_id": int(gts[i]),
            "pred": pred,
            "gt": gt
        })
        
        if gt in close_ans:
            yn_total += 1
            if preds[i] == gts[i]:
                yn_acc += 1
        else:
            other_total += 1
            if preds[i] == gts[i]:
                other_acc += 1 
    acc["yn"] = yn_acc / yn_total *100
    acc["other"] = other_acc / other_total *100
    
    return result, acc

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

def train3(config, model, data_loader, optimizer, loss_fn, epoch, device, train_size, id2ans):
    model.train()
    train_loss = []
    close_ans = ['yes', 'no', 'Yes', 'No']
    preds = []
    gts = []
    for i, (qid, image, tokens, segment_ids, input_mask, answer, weights) in enumerate(data_loader):
        # qid: list  image: torch[8, 3, 480, 480] tokens: torch[8,20] 
        # segment_ids: torch[8,20]  input_mask: torch[8,20] answer: torch[8]  weights: torch[8] 
        alpha = config['alpha']*min(1, i/len(data_loader)) 
        image = image_enhance(image)
        image, answer = image.to(device), answer.to(device) # bs*3*480*480 
        tokens, segment_ids, input_mask = tokens.to(device), segment_ids.to(device), input_mask.to(device)
        # ans_inputs = ans_inputs.to(device)
        logits, loss_ita = model(image, tokens, segment_ids, input_mask, alpha=alpha, training=True)
        
        loss_ce = loss_fn(logits, answer)
        # loss_mse = MSE_loss(hid_emb, ans_embeds)
        # loss = config["alpha"]*loss_mse + loss_ce
        loss = loss_ce + loss_ita
        loss_np = loss.detach().cpu().numpy()      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("\r[epoch: %d][%d/%d]" % (epoch, i, train_size/config["batch_size_train"]), end="        ")

        pred = logits.argmax(1).detach()
        gt = answer.detach()
        preds.append(pred)
        gts.append(gt)
        train_loss.append(float(loss_np))
    
    preds = torch.cat(preds).cpu().numpy()
    gts = torch.cat(gts).cpu().numpy()
    train_loss = np.mean(train_loss)

    total_acc = (preds == gts).mean() * 100.
    acc = {'total_acc': np.round(total_acc, 4)}
    
    yn_acc = 0
    other_acc = 0
    yn_total = 0.0001
    other_total = 0.0001
    for i in range(len(preds)):
        if id2ans[gts[i]] in close_ans:
            yn_total += 1
            if preds[i] == gts[i]:
                yn_acc += 1
        else:
            other_total += 1
            if preds[i] == gts[i]:
                other_acc += 1 
    acc["yn"] = yn_acc / yn_total *100
    acc["other"] = other_acc / other_total *100

    return acc, train_loss

def validate3(config, model, data_loader, optimizer, loss_fn, epoch, device, val_size, idx2ans):
    model.eval()
    val_loss = []
    preds = []
    gts =[]
    qids = []

    bar = tqdm(data_loader, leave=False)
    with torch.no_grad():
        for (qid, image, tokens, segment_ids, input_mask, answer, weights) in bar:
            image, answer = image.to(device), answer.to(device)
            tokens, segment_ids, input_mask = tokens.to(device), segment_ids.to(device), input_mask.to(device)
            logits = model(image, tokens, segment_ids, input_mask, not_pretrain=True)
            loss = loss_fn(logits, answer)

            loss_np = loss.detach().cpu().numpy()
            pred = logits.argmax(1).detach()
            gt = answer.detach()
            
            preds.append(pred)
            gts.append(gt)
            for i in qid:
                qids.append(i)
            val_loss.append(loss_np)

        val_loss = np.mean(val_loss)

    preds = torch.cat(preds).cpu().numpy()
    gts = torch.cat(gts).cpu().numpy()

    result, acc = get_acc(preds, gts, qids, idx2ans)

    return val_loss, result, acc

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
    # id2ans = json.load(open(os.path.join(config['vqa_root'], 'en_data/ids2ans.json')))
    id2ans = json.load(open(os.path.join(config['vqa_root'], config['label2ans'])))
    ans_embeds_all = []
    # ans_embeds = json.load(open(os.path.join(config['vqa_root'], config['ans_embeds'])))[0]
    # for ans, embeds in ans_embeds.items():
    #     ans_embeds_all.append(embeds)
    # ans_embeds_all = torch.tensor(ans_embeds_all)

    # loading dataset
    print("Creating medical vqa datasets")
    datasets = create_dataset(config=config)
    ans_size = len(id2ans)
    train_size, val_size, test_size = datasets[0].ann_size, datasets[1].ann_size, datasets[2].ann_size

    train_loader, val_loader, test_loader = create_loader(datasets,[None, None, None],
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4,4,4], is_trains=[True, False, False]) 
    
    if config["use_CPT"]:
        CPT_dataset = create_CPT_dataset(config)
        CPT_loader = DataLoader(CPT_dataset,batch_size=config["CPT_size"],num_workers=1,pin_memory=True,
                                sampler=None, shuffle=True, drop_last=True)    
        CPT_dataset_size = CPT_dataset.ann_size
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
    else:
        model = BiVision_VQA(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)

    if args.load_checkpoint:
        pretrained_dict = torch.load(args.load_checkpoint)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fusion_layer' not in k and "classifier.2" not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(torch.load(args.load_checkpoint), strict=False)
    
    model.to(args.device)

    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])
    loss_fn = nn.CrossEntropyLoss()
    # MSE_loss = nn.MSELoss()
    # SCL_loss = SupConLoss(device=args.device)

    best_acc = 0
    best_epoch = 0
    print("Start training")
    start_time = 0

    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            acc, train_loss = train3(config, model, train_loader, optimizer, loss_fn, epoch, device, \
                train_size, id2ans)
            print("epoch: %d," % epoch, end=" ")
            print(acc, end="\n    ")

            val_loss, result, val_acc = validate3(config, model, val_loader, optimizer, 
                                                loss_fn, epoch, device, val_size, id2ans)
            print("epoch: %d," % epoch, end=" ")
            print(val_acc, end="    ")
            result_file = os.path.join(args.result_dir, 'result_val_%d_%d_%d.json' % (seed, epoch, rand_a))
            json.dump(result, open(result_file, 'w'))
            print("save to %s" % result_file)

            if val_acc['val_acc'] > best_acc:
                model_path = os.path.join(args.save_dir, args.dataset+"_"+str(seed)+"_best.pt")
                torch.save(model.state_dict(), model_path)
                print("save model to %s" % model_path)
                best_acc = val_acc['val_acc']
            
            log_stats = {
                        "seed": int(args.seed),
                        "model": args.model,
                        "datatime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                        "epoch": epoch,
                        "train_acc": acc,
                        "train_loss": float(train_loss),
                        "val_acc": val_acc,
                        "val_loss": float(val_loss)
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n") 
        else:
            break
    test_loss, result, test_acc = validate3(config, model, test_loader, optimizer, 
                                                        loss_fn, epoch, device, test_size, id2ans)
    print("test_acc: ")
    print(test_acc)
    result_file = os.path.join(args.result_dir, 'result_test_%d.json' % (seed))
    json.dump(result, open(result_file, 'w'))
    print("save to %s" % result_file)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/PathVQA.yaml', type=str)
    parser.add_argument('--output_dir', default='output/PathVQA', type=str)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=51, type=int)
    parser.add_argument('--load_checkpoint', 
                        # default="ckpt/PathVQA_58_best.pt",  # {"total_acc": 88.3486, "yn": 90.48452478198922, "other": 85.64026013025462}
                        # default="ckpt/VQA_RAD_59_best.pt",   # {"val_acc": 61.1973, "yn": 74.50169521236967, "other": 44.49977750111249}
                        # default=None,
                        # default="ckpt/VQA_SLAKE_57_best.pt", # {'total_acc': 97.0277, 'yn': 97.85586075903151, 'other': 96.59758439228008} {'val_acc': 75.7776, 'yn': 84.50680420618534, 'other': 71.3880008668543}
                        # 56 slake {'val_acc': 75.1178, 'yn': 81.68991073264583, 'other': 71.81292944344271}
                        # default="ckpt/VQA_RAD_51_best68.pt",
                        # default="ckpt/VQA_SLAKE_52_pretrain.pt",
                        type=str)
    parser.add_argument('--save_dir', default="ckpt", type=str)
    parser.add_argument('--dataset', default="PathVQA", type=str)
    parser.add_argument('--model', default='vqa16', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)

