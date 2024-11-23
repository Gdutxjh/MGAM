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
from data.retrieval_dataset import slake_collate_fn
from models.retrieval_vqa import retrieval_vqa

from utils import cosine_lr_schedule, calculate_bleu_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

def get_acc(result):
    # {
    #     "qid": qid[k],
    #     "question": question[k],
    #     "pred": int(pred[k].detach().cpu()),
    #     "preds_top": pred_topk_ids,
    #     "answer": answer[k],
    #     "answer_id": int(answer_id[k].detach().cpu()),
    #     "answer_type": answer_type[k]
    # }
    m = len(result)
    acc = {"CLOSED": 0, "OPEN": 0}
    acc_top = {"CLOSED": 0, "OPEN": 0}
    total_acc = {"CLOSED": 1e-5, "OPEN": 1e-5}
    
    for i in range(m):
        pred = result[i]["pred"]
        preds_top = result[i]["preds_top"]
        answer_id = result[i]["answer_id"]
        answer_type = result[i]["answer_type"]
        
        total_acc[answer_type] +=1
        if pred == answer_id:
            acc[answer_type] +=1
        if answer_id in preds_top:
            acc_top[answer_type] +=1
    acc_close = acc["CLOSED"]/total_acc["CLOSED"]
    top_acc_close = acc_top["CLOSED"]/total_acc["CLOSED"]
    acc_open = acc["OPEN"]/total_acc["OPEN"]
    top_acc_open = acc_top["OPEN"]/total_acc["OPEN"]
    acc = (acc["CLOSED"]+acc["OPEN"])/(total_acc["CLOSED"]+total_acc["OPEN"])
    top_acc = (acc_top["CLOSED"]+acc_top["OPEN"])/(total_acc["CLOSED"]+total_acc["OPEN"])
    print("acc_close: %4f, acc_open: %4f, top_acc_close: %4f, top_acc_open: %4f" %(acc_close, acc_open, top_acc_close, top_acc_open))
    print("total_acc: %4f, total_top_acc: %4f" % (acc, top_acc))
    return acc

def train3(config, model, data_loader, optimizer, epoch, device, train_size, id2ans):
    model.train()
    train_loss = []
 
    result = []
    for i, (qid, image, question, answer, answer_type, answer_id, retrieval_text) in enumerate(data_loader):
        # qid: list  image: torch[8, 3, 480, 480] tokens: torch[8,20] 
        # segment_ids: torch[8,20]  input_mask: torch[8,20] answer: torch[8]  weights: torch[8] 

        image, answer_id = image.to(device), answer_id.to(device) # bs*3*480*480 

        logits, loss = model(image, question, retrieval_text, answer_id)

        loss_np = loss.detach().cpu().numpy()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("\r[epoch: %d][%d/%d] loss: %f" % (epoch, i, train_size/config["batch_size_train"], loss_np), end="        ")

        pred = logits.argmax(1).detach()
        for k in range(len(logits)):
            pred_topk_ids = logits[k].argsort()[-5:].detach().cpu().tolist()
        
            result.append({
                "qid": qid[k],
                "question": question[k],
                "pred": int(pred[k].detach().cpu()),
                "preds_top": pred_topk_ids,
                "answer": answer[k],
                "answer_id": int(answer_id[k].detach().cpu()),
                "answer_type": answer_type[k]
            })
        
        train_loss.append(float(loss_np))
    train_loss
    # model.save_ans_embeds()
    acc = get_acc(result)
    return acc, train_loss

def validate3(config, model, data_loader, optimizer, epoch, device, val_size, idx2ans):
    model.eval()
    val_loss = []
    result = []

    bar = tqdm(data_loader, leave=False)

    with torch.no_grad():
        for (qid, image, question, answer, answer_type, answer_id, retrieval_text) in bar:
            image, answer_id = image.to(device), answer_id.to(device)
            logits, loss = model(image, question, retrieval_text, answer_id)
            
            loss_np = loss.detach().cpu().numpy()
            pred = logits.argmax(1).detach()
            
            for k in range(len(logits)):
                pred_topk_ids = logits[k].argsort()[-3:].detach().cpu().tolist()
                
                result.append({
                    "qid": qid[k],
                    "question": question[k],
                    "pred": int(pred[k].detach().cpu()),
                    "preds_top": pred_topk_ids,
                    "answer": answer[k],
                    "answer_id": int(answer_id[k].detach().cpu()),
                    "answer_type": answer_type[k]
                })
            val_loss.append(loss_np)
        # val_loss = np.mean(val_loss)

    acc = get_acc(result)
    return val_loss, result, acc

def main(args, config):
    device = args.device
    
    # seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # id2ans = json.load(open(os.path.join(config['vqa_root'], 'en_data/ids2ans.json')))
    id2ans = json.load(open(os.path.join(config['vqa_root'], config['label2ans'])))

    # loading dataset
    print("Creating medical vqa datasets")
    datasets = create_dataset(config=config)
    ans_size = len(id2ans)
    train_size, val_size, test_size = datasets[0].ann_size, datasets[1].ann_size, datasets[2].ann_size

    train_loader, val_loader, test_loader = create_loader(datasets,[None, None, None],
                                        batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                        num_workers=[4,4,4], is_trains=[True, False, False], collate_fns=[slake_collate_fn, slake_collate_fn, slake_collate_fn]) 
    
    RN50_cfg = json.load(open(config["RN50_path"]))
    text_cfg = RN50_cfg["text_cfg"]
    
    # Model
    print("Creating model")
    model = retrieval_vqa(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device, config=config, text_cfg=text_cfg)
    
    if args.load_checkpoint:
        pretrained_dict = torch.load(args.load_checkpoint)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)} #  and k.split(".")[0] not in layers_out)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    
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
            acc, train_loss = train3(config, model, train_loader, optimizer, epoch, device, train_size, id2ans)
            print("epoch: %d," % epoch, end=" ")
            print(acc, end="\n    ")

            print("model: %s seed: %d" % (args.model, args.seed), end="  ")
            val_loss, result, val_acc = validate3(config, model, val_loader, optimizer, 
                                                epoch, device, val_size, id2ans)
            print("epoch: %d," % epoch, end=" ")
            print(val_acc, end="    ")
            # result_file = os.path.join(args.result_dir, 'result_val_%d_%d_%d.json' % (seed, epoch, rand_a))
            # json.dump(result, open(result_file, 'w'))
            # print("save to %s" % result_file)

            if val_acc > best_acc:
                model_path = os.path.join(args.save_dir, config["dataset"]+"_"+str(seed)+"_best.pt")
                torch.save(model.state_dict(), model_path)
                print("save model to %s" % model_path)
                best_acc = val_acc
            
            log_stats = {
                        "seed": int(args.seed),
                        "model": args.model,
                        "datatime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                        "epoch": epoch,
                        "train_acc": acc,
                        "train_loss": float(sum(train_loss)/len(train_loss)),
                        "val_acc": val_acc,
                        "val_loss": float(sum(val_loss)/len(val_loss)),
                        "seed": seed
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n") 
        else:
            break
    test_loss, result, test_acc = validate3(config, model, test_loader, optimizer, 
                                                epoch, device, test_size, id2ans)
    print("model: %s seed: %d" % (args.model, args.seed), end="  ")
    print("test_acc: ")
    print(test_acc)
    result_file = os.path.join(args.result_dir, 'result_test_%d.json' % (seed))
    json.dump(result, open(result_file, 'w'), indent=2)
    print("save to %s" % result_file)

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/retrieval/SLAKE_retrieval.yaml', type=str)   # PathVQA
    parser.add_argument('--output_dir', default='output/retrieval/SLAKE_retrieval', type=str)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=1, type=int) # 26 only sentence bert
    parser.add_argument('--load_checkpoint', 
                        # default="ckpt/PathVQA_33_pretrain.pt", # 23
                        type=str)
    parser.add_argument('--save_dir', default="ckpt/Slake", type=str)
    # parser.add_argument('--dataset', default="VQA_RAD", type=str)
    parser.add_argument('--model', default='retrieval_vqa', type=str)    # 8
    args = parser.parse_args()
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)

