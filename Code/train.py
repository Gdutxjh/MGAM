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

from data import create_dataset, create_loader
from data.PathVQA_dataset import vqa_collate_fn
from models.bivi_vqa import BiVision_VQA
from models.vqa3 import BiVision_VQA3, BiVision_VQA8
from model import Bistep_vqa_new
# from models.vqa13 import Bistep_vqa, Bistep_vqa2
from models.vqa14 import Bistep_vqa7, Bistep_vqa8, Bistep_vqa9
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

def get_acc(preds, gts, qids, idx2ans, preds_top, ques_list):
    total_acc = (preds == gts).mean() * 100.
    # total_bleu = calculate_bleu_score(preds, gts, idx2ans)
    # np.round(total_acc, 4) 四舍五入到小数点后四位
    acc = {'val_acc': np.round(total_acc, 4)}
    close_ans = ['yes', 'no', 'Yes', 'No']
    # bleu = {'total_bleu': np.round(total_bleu, 4)}
    
    result = []
    result_false = []
    yn_acc = 0
    other_acc = 0
    yn_top = 0
    other_top = 0
    yn_total = 0.001
    other_total = 0.001
    for i in range(len(preds)):
        pred = idx2ans[int(preds[i])]
        gt = idx2ans[int(gts[i])]
        top_pred = []
        for idx in preds_top:
            top_pred.append(idx)
        
        result.append({
            "qid": qids[i],
            "question": ques_list[i],
            "pred_id": int(preds[i]),
            "gt_id": int(gts[i]),
            "pred": pred,
            "gt": gt,
            "top_pred": [idx2ans[x] for x in preds_top[i]]
        })
        
        if preds[i] != gts[i]:
            result_false.append({
                "qid": qids[i],
                "pred_id": int(preds[i]),
                "gt_id": int(gts[i]),
                "pred": pred,
                "gt": gt,
                "top_pred": [idx2ans[x] for x in preds_top[i]]
            })
        
        if gt in close_ans:
            yn_total += 1
            if preds[i] == gts[i]:
                yn_acc += 1
            if gts[i] in preds_top[i]:
                yn_top += 1
        else:
            other_total += 1
            if preds[i] == gts[i]:
                other_acc += 1 
            if gts[i] in preds_top[i]:
                other_top += 1
    acc["yn"] = yn_acc / yn_total *100
    acc["other"] = other_acc / other_total *100
    acc["yn_top"] = yn_top / yn_total *100
    acc["other_top"] = other_top / other_total *100

    return result, result_false, acc

def train3(config, model, data_loader, optimizer, loss_fn, epoch, device, train_size, id2ans, ans_tokens_list):
    model.train()
    train_loss = []
    close_ans = ['yes', 'no', 'Yes', 'No']
    preds = []
    gts = []
    preds_top = []
    for i, (qid, image, tokens, segment_ids, input_mask, answer, question, ques_target) in enumerate(data_loader):
        # qid: list  image: torch[8, 3, 480, 480] tokens: torch[8,20]  "" ,
        # segment_ids: torch[8,20]  input_mask: torch[8,20] answer: torch[8]  weights: torch[8] 

        # image = image_enhance(image)
        image, answer, ques_target = image.to(device), answer.to(device), ques_target.to(device) # bs*3*480*480 
        tokens, segment_ids, input_mask = tokens.to(device), segment_ids.to(device), input_mask.to(device)

        # logits, contrastive_loss, loss_triplet, loss_ce2, loss_ce1 = model(image, tokens, segment_ids, input_mask, answer, id2ans, ans_tokens_list)
        # logits, contrastive_loss, loss_triplet, loss_ce = model(image, tokens, segment_ids, input_mask, answer, id2ans, ans_tokens_list, question=question)
        logits, loss = model(image, tokens, segment_ids, input_mask, answer, id2ans, ans_tokens_list, question=question, ques_target=ques_target)
        # logits, loss_ce, loss_triplet = model(image, tokens, segment_ids, input_mask, answer, id2ans, ans_tokens_list, question=question)

        # loss = loss_fn(logits, answer)
        # loss = loss_ce1 + loss_ce2 + contrastive_loss + loss_triplet
        # loss = loss_ce + loss_triplet

        # 将loss张量从计算图中分离出来，并将其转移到CPU上，然后转换为NumPy数组，以便我们可以打印出来。
        loss_np = loss.detach().cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 

        print("\r[epoch: %d][%d/%d] loss: %f" % (epoch, i, train_size/config["batch_size_train"], loss_np), end="        ")

        # 从logits中获取每个样本的预测类别索引，并将其从计算图中分离出来
        pred = logits.argmax(1).detach()
        # 获得每个样本的前两个预测类别索引，并将其添加到preds_top列表中
        for k in range(len(logits)):
            pred_topk_ids = logits[k].argsort()[-2:].detach().cpu().tolist()
            preds_top.append(pred_topk_ids)
        # 将真实标签answer从计算图中分离出来
        gt = answer.detach()
        # 将预测标签pred和真实标签gt添加到preds和gts列表中
        preds.append(pred)
        gts.append(gt)
        # 将当前损失loss_np添加到train_loss列表中
        train_loss.append(float(loss_np))
    
    # model.save_ans_embeds()
    # 将训练过程中收集的预测值和真实值从多个批次中拼接起来，并计算训练损失的平均值
    preds = torch.cat(preds).cpu().numpy()
    gts = torch.cat(gts).cpu().numpy()
    train_loss = np.mean(train_loss)

    total_acc = (preds == gts).mean() * 100.
    acc = {'total_acc': np.round(total_acc, 4)}
    
    yn_acc = 0
    other_acc = 0
    yn_top = 0
    other_top = 0
    yn_total = 0.0001
    other_total = 0.0001
    for i in range(len(preds)):
        if id2ans[gts[i]] in close_ans:
            yn_total += 1
            if preds[i] == gts[i]:
                yn_acc += 1
            if gts[i] in preds_top[i]:
                yn_top += 1
        else:
            other_total += 1
            if preds[i] == gts[i]:
                other_acc += 1 
            if gts[i] in preds_top[i]:
                other_top += 1
    acc["yn"] = yn_acc / yn_total *100
    acc["other"] = other_acc / other_total *100
    acc["yn_top"] = yn_top / yn_total *100
    acc["other_top"] = other_top / other_total *100

    return acc, train_loss

def validate3(config, model, data_loader, optimizer, loss_fn, epoch, device, val_size, idx2ans, ans_tokens_list):
    model.eval()
    val_loss = []

    preds = []
    gts =[]
    qids = []
    ques_list = []
    preds_top = []
    bar = tqdm(data_loader, leave=False)

    with torch.no_grad():
        for (qid, image, tokens, segment_ids, input_mask, answer, question, ques_target) in bar:
            image, answer, ques_target = image.to(device), answer.to(device), ques_target.to(device)
            tokens, segment_ids, input_mask = tokens.to(device), segment_ids.to(device), input_mask.to(device)
            logits, _ = model(image, tokens, segment_ids, input_mask, answer, idx2ans, ans_tokens_list, 
                           question=question, ques_target=ques_target, val=True)
            loss = loss_fn(logits, answer)
            
            loss_np = loss.detach().cpu().numpy()   
            pred = logits.argmax(1).detach()    # 
            gt = answer.detach()

            preds.append(pred)
            gts.append(gt)
            for k in range(len(logits)):
                pred_topk_ids = logits[k].argsort()[-3:].detach().cpu().tolist()
                preds_top.append(pred_topk_ids)
                ques_list.append(question[k])
                
            if not isinstance(qid,list):
                qid = qid.tolist()
            for i in qid:
                qids.append(i)
            val_loss.append(loss_np)

        val_loss = np.mean(val_loss)

    preds = torch.cat(preds).cpu().numpy()
    gts = torch.cat(gts).cpu().numpy()
    
    result, result_false, acc = get_acc(preds, gts, qids, idx2ans, preds_top, ques_list)
    return val_loss, result, acc, result_false

def main(args, config):
    device = args.device
    torch.backends.cudnn.enabled = False
    config["pretrain"] = False
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
    
    # ans_embeds_all = torch.load(config["answer_embeds_path"])
    # ans_embeds_all = ans_embeds_all.to(device)
    # ans_tokens_list = json.load(open(os.path.join(config['vqa_root'], config['ans_tokens']), "r"))
    ans_tokens_list = None
    # id2ans = json.load(open(os.path.join(config['vqa_root'], 'en_data/ids2ans.json')))
    id2ans = json.load(open(os.path.join(config['vqa_root'], config['label2ans'])))

    # loading dataset
    print("Creating medical vqa datasets")
    datasets = create_dataset(config=config)
    ans_size = len(id2ans)
    train_size, val_size, test_size = datasets[0].ann_size, datasets[1].ann_size, datasets[2].ann_size

    train_loader, val_loader, test_loader = create_loader(datasets, [None, None, None],
                                                          collate_fns=[None, None, None],  # 指定collate_fns参数
                                                          batch_size=[config['batch_size_train'],
                                                                      config['batch_size_test'],
                                                                      config['batch_size_test']],
                                                          num_workers=[4, 4, 4], is_trains=[True, False, False])

    # train_loader, val_loader, test_loader = create_loader(datasets,[None, None, None],
    #                                           batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
    #                                           num_workers=[4,4,4], is_trains=[True, False, False])
    
    # Model
    print("Creating model")
    
    if args.model == 'vqa18':
        model = Bistep_vqa8(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device, config=config)
    elif args.model == 'vqa19':
        model = Bistep_vqa9(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device, config=config)
    elif args.model == 'model':
        model = Bistep_vqa_new(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    else:
        model = BiVision_VQA(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)

    # layers_out = ["classifier1.2.weight", "classifier1.2.bias", "classifier2.2.weight", "classifier2.2.bias"]
    # layers_out = ["classifier2", "gobal_attention", "classifier1"]
    # layers_out = ["local_attention.w", "gobal_attention.w"]
    # layers_out = ["sentence_bert.0.auto_model.embeddings.word_embeddings.weight"]
    
    if args.load_checkpoint:
        pretrained_dict = torch.load(args.load_checkpoint, map_location='cpu')
        model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fusion_layer' not in k and "classifier.2" not in k)}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and "classifier.2" not in k)}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)} #  and k.split(".")[0] not in layers_out)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
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
            acc, train_loss = train3(config, model, train_loader, optimizer, loss_fn, epoch, device, train_size, \
                id2ans, ans_tokens_list)
            print("epoch: %d," % epoch, end=" ")
            print(acc, end="\n    ")

            print("model: %s seed: %d" % (args.model, args.seed), end="  ")
            val_loss, result, val_acc, result_false = validate3(config, model, val_loader, optimizer, 
                                                loss_fn, epoch, device, val_size, id2ans, ans_tokens_list)
            print("epoch: %d," % epoch, end=" ")
            print(val_acc)
            # result_file = os.path.join(args.result_dir, 'result_val_%d_%d_%d.json' % (seed, epoch, rand_a))
            # json.dump(result, open(result_file, 'w'))
            # print("save to %s" % result_file)

            if val_acc['val_acc'] > best_acc:
                model_path = os.path.join(args.save_dir, args.dataset+"_"+str(seed)+"_best.pt")
                torch.save(model.state_dict(), model_path)
                print("save model to %s" % model_path)
                best_acc = val_acc['val_acc']
                
                result_file = os.path.join(args.result_dir, 'result_val_%d.json' % (seed))
                json.dump(result, open(result_file, 'w'), indent=2)
                print("save to %s" % result_file)
            
            log_stats = {
                        "seed": int(args.seed),
                        "model": args.model,
                        "datatime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                        "epoch": epoch,
                        "train_acc": acc,
                        "train_loss": float(train_loss),
                        "val_acc": val_acc,
                        "val_loss": float(val_loss),
                        "seed": seed
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n") 
        else:
            break
    test_loss, test_result, test_acc, result_false = validate3(config, model, test_loader, optimizer, 
                                                loss_fn, 1, device, test_size, id2ans, ans_tokens_list)
    print("model: %s seed: %d" % (args.model, args.seed), end="  ")
    print("test_acc: ")
    print(test_acc)
    result_file = os.path.join(args.result_dir, 'result_test_%d.json' % (seed))
    json.dump(test_result, open(result_file, 'w'), indent=2)
    print("save to %s" % result_file)
    
    result_file = os.path.join(args.result_dir, 'result_test_%d_false.json' % (seed))
    json.dump(result_false, open(result_file, 'w'), indent=2)
    print("save to %s" % result_file)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/VQA_RAD.yaml', type=str)   # PathVQA
    parser.add_argument('--output_dir', default='output/VQA_RAD_test', type=str)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--seed', default=53, type=int) # 26 only sentence bert
    parser.add_argument('--load_checkpoint', 
                        # default="ckpt/PathVQA_30_54.pt", # 23
                        # default="ckpt/VQA_RAD_30_two_mask_74.pt",
                        # default="ckpt/VQA_RAD_29_best.pt", # 23
                        # VQA_SLAKE_46 'val_acc': 7.2856, 'yn': 85.9152509429551, 'other': 72.94607231434517
                        # default="ckpt/PathVQA_29_best.pt",
                        # default="ckpt/pretrain_PathVQA_29_bert.pt",
                        # default="ckpt/VQA_SLAKE_28_best.pt",
                        # default="ckpt/VQA_RAD_53_best.pt",
                        # default = "ckpt/VQA_RAD_28_74_18.pt",
                        type=str)
    parser.add_argument('--save_dir', default="ckpt", type=str)
    parser.add_argument('--dataset', default="VQA_RAD", type=str)
    parser.add_argument('--model', default='model', type=str)    # 8
    args = parser.parse_args()
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'


    yaml = yaml.YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True) 
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)

