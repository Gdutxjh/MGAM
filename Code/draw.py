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
from transformers import BertTokenizer

from data import create_dataset, create_loader
from data.PathVQA_dataset import vqa_collate_fn
from models.bivi_vqa import BiVision_VQA
from model import Bistep_vqa_new
from models.vqa3 import BiVision_VQA3, BiVision_VQA8
from models.vqa13 import Bistep_vqa, Bistep_vqa2
from models.vqa14 import Bistep_vqa7, Bistep_vqa8, Bistep_vqa9
import matplotlib.pyplot as plt

def plot_attention(img, qid, question, anwer, pred, attention):
    n_heads=attention.shape[0]

    plt.figure(figsize=(10, 10))
    # text= ["Original Image", "Head Mean"]
    # for i, fig in enumerate([img, np.mean(attention, 0)]):
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(fig, cmap='inferno')
    #     # plt.title(text[i])
    # plt.show()
    text = str(qid) + "\n" + str(question) + "\n" + str(pred) + "\n" + str(anwer)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='inferno')
    plt.imshow(np.mean(attention, 0), cmap='inferno', alpha=0.5)
    # plt.title(text)
    plt.savefig('attn_output/VQA_RAD/'+str(text)+"_attn.jpg")

    # plt.figure(figsize=(10, 10))
    # for i in range(n_heads):
    #     plt.subplot(n_heads//3, 3, i+1)
    #     plt.imshow(attention[i], cmap='inferno')
    #     plt.title(f"Head n: {i+1}")
    # plt.tight_layout()
    # plt.show()

def draw_image1(config, image, qid, question, answer, pred, score):
    patch_size = 36
    w_featmap=image.shape[-2] // patch_size    
    h_featmap=image.shape[-1] // patch_size 
    nh = 12
    attentions = score[:, 0, 1:37]
    attentions=attentions.reshape(nh, w_featmap, h_featmap)
    attentions=nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size , mode="nearest")[0].cpu().numpy()
    
    im2show = np.copy(image.permute(1,2,0).cpu().numpy())
    plt.figure(figsize=(10, 10))
    text = str(qid.numpy()) + "\n" + str(question) + "\n" + str(pred) + "\n" + str(answer)
    plt.subplot(1, 4, 1)
    plt.imshow(im2show, cmap='inferno', alpha=0.8)
    plt.imshow(np.mean(attentions, 0), cmap='inferno', alpha=0.4)
    # plt.title(text)
    plt.savefig('attn_output/VQA_RAD/'+text+"_attn.jpg")

# cat
def draw_image2(config, image, qid, question, answer, pred, score, path):
    patch_size = 36
    w_featmap=image.shape[-2] // patch_size    
    h_featmap=image.shape[-1] // patch_size 
    nh = 12
    save_name = ["last", "coarse", "fine"]
    text = str(qid) + "__" + str(question) + "__" + str(answer)
    output_path = 'attn_output/PathVQA/'+path+'/'+text
    output_path = output_path +"__" + str(pred) + ".jpg" 
    if os.path.exists(output_path) or len(output_path) > 200:
        return
    
    plt.figure(figsize=(40, 10))
    im2show = np.copy(image.permute(1,2,0).cpu().numpy())
    plt.subplot(1, 4, 1)
    plt.imshow(im2show, cmap='inferno')
    plt.axis('off')
    # text = str(qid.numpy()) + "__" + str(question) + "__" + str(pred) + "__" + str(answer)
    # text = str(qid.numpy()) + "__" + str(question) + "__" + str(answer)
    # output_path = 'attn_output/VQA_RAD/'+path+'/'+text
    
    
    for i in range(len(score)):
        attentions = score[i, :, 1:]
        attentions=attentions.reshape(nh, w_featmap, h_featmap)
        attentions=nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size , mode="nearest")[0].cpu().numpy()
        
        plt.subplot(1, 4, i+2)
        plt.imshow(im2show, cmap='inferno', alpha=0.8)
        plt.imshow(np.mean(attentions, 0), cmap='inferno', alpha=0.4)
        plt.axis('off')
        # plt.title(text)
    plt.savefig(output_path)

# MFB 
def draw_image4(config, image, qid, question, answer, pred, score):
    # socre  [8, 57, 20]
    patch_size = 36
    w_featmap=image.shape[-2] // patch_size    
    h_featmap=image.shape[-1] // patch_size

    nh = score.shape[0]
    save_name = ["last", "coarse", "fine"]
    
    plt.figure(figsize=(10, 10))
    im2show = np.copy(image.permute(1,2,0).cpu().numpy())
    plt.subplot(1, 1, 1)
    plt.imshow(im2show, cmap='inferno')
    plt.axis('off')
    # text = str(qid.numpy()) + "__" + str(question) + "__" + str(pred) + "__" + str(answer)
    text = str(qid) + "__" + str(question) + "__" + str(answer)
    output_path = 'attn_output/cls/BAN/'+text
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    plt.savefig(output_path+'/pred__'+str(pred)+".jpg")
    # for i in range(len(score)):
    plt.figure(figsize=(10, 10))
    attentions = score[:, 1:-20, 0]
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size , mode="nearest")[0].cpu().numpy()
    
    plt.subplot(1, 1, 1)
    plt.imshow(im2show, cmap='inferno', alpha=0.8)
    plt.imshow(np.mean(attentions, 0), cmap='inferno', alpha=0.4)
    plt.axis('off')
    # plt.title(text)
    plt.savefig(output_path+'/'+save_name[0]+".jpg")


def draw_image3(config, image, qid, question, answer, pred, score):
    # socre  [8, 57, 20]
    # score  torch.Size([3, 2, 17]) VQA_RAD
    patch_size = 36
    w_featmap = image.shape[-2] // patch_size
    h_featmap = image.shape[-1] // patch_size

    nh = score.shape[0]
    save_name = ["last", "coarse", "fine"]

    plt.figure(figsize=(10, 10))
    im2show = np.copy(image.permute(1, 2, 0).cpu().numpy())
    plt.subplot(1, 1, 1)
    plt.imshow(im2show, cmap='inferno')
    plt.axis('off')
    # text = str(qid.numpy()) + "__" + str(question) + "__" + str(pred) + "__" + str(answer)
    text = str(qid) + "__" + str(question) + "__" + str(answer)
    output_path = 'attn_output/cls/BAN/' + text
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    plt.savefig(output_path + '/pred__' + str(pred) + ".jpg")
    # for i in range(len(score)):
    plt.figure(figsize=(10, 10))
    attentions = score[:, 0, 1:37]
    # attentions = score[:, 1:-20, 0]
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
        0].cpu().numpy()

    plt.subplot(1, 1, 1)
    plt.imshow(im2show, cmap='inferno', alpha=0.8)
    plt.imshow(np.mean(attentions, 0), cmap='inferno', alpha=0.4)
    plt.axis('off')
    # plt.title(text)
    plt.savefig(output_path + '/' + save_name[0] + ".jpg")


def draw_word(config, image, qid, question, answer, pred, score, path, tokens, input_mask, tokenizer):
    # socre  [12, 20, 57]
    patch_size = 36
    w_featmap=image.shape[-2] // patch_size    
    h_featmap=image.shape[-1] // patch_size 
    nh = score.shape[0]
    
    ques_token = tokenizer.decode(tokens)
    other_tokens = [101, 102, 0]
    new_tokens = [tokenizer.decode([x]) for x in tokens if x not in other_tokens]
    
    # 
    # text = str(qid.numpy()) + "__" + str(question) + "__" + str(pred) + "__" + str(answer)
    text = str(qid.numpy()) + "__" + str(question) + "__" + str(answer)
    output_path = 'attn_output/cls/VQA_RAD/word_fine/'+text
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    # plt.figure(figsize=(10, 10))
    plt.figure()
    # fine
    attentions = score[0, 1:-20, -20:]
    attentions = attentions[:, 1:len(new_tokens)+1].cpu().numpy().T
    # score2
    # attentions = score[0, :, 1:-20]
    # attentions = F.normalize(attentions[1:len(new_tokens)+1, :]).cpu().numpy()
    # attentions = attentions[1:len(new_tokens)+1, :].cpu().numpy()
    # attentions=attentions.reshape(nh, w_featmap, h_featmap)
    # attentions=nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=5 , mode="nearest")[0].cpu().numpy()
    plt.subplot(1, 1, 1)
    # fine
    # plt.imshow(np.mean(attentions, 0).T, cmap='inferno', alpha=1)
    # score2
    plt.imshow(attentions, cmap='inferno', alpha=1)
    # fine
    # plt.yticks([i for i in range(len(attentions[0][0]))], new_tokens)
    plt.yticks([i for i in range(len(new_tokens))], new_tokens)
    # plt.colorbar()
    # plt.axis('off')
    # plt.title(text)
    plt.savefig(output_path+"/word-image.jpg")
    
    
    
    
    # plt.figure(figsize=(10, 10))
    plt.figure()
    # fine
    # attentions = F.normalize(score[:2, -19:-(20-len(new_tokens)-1), -19:-(20-len(new_tokens)-1)]).cpu().numpy()
    attentions = score[:2, -19:-(20-len(new_tokens)-1), -19:-(20-len(new_tokens)-1)].cpu().numpy()
    # score2
    # attentions = F.normalize(score[:2, 1:len(new_tokens)+1, -19:-(20-len(new_tokens)-1)]).cpu().numpy()
    # attentions=attentions.reshape(nh, w_featmap, h_featmap)
    # attentions=nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=5 , mode="nearest")[0].cpu().numpy()
    plt.subplot(1, 1, 1)
    plt.imshow(np.mean(attentions, 0), cmap='inferno', alpha=1)
    # plt.yticks([i for i in range(len(attentions[0][0]))], new_tokens)
    # plt.xticks([i for i in range(len(attentions[0][0]))], new_tokens)
    plt.yticks([i for i in range(len(new_tokens))], new_tokens)
    plt.xticks([i for i in range(len(new_tokens))], new_tokens)
    # plt.colorbar()
    # plt.axis('off')
    # plt.title(text)
    plt.savefig(output_path+"/word-word.jpg")

# split
def draw_image(config, image, qid, question, answer, pred, score):
    # number = ["test_0042", "test_0065", "test_0092", "test_0167", "test_0273", "test_0312", "test_0886"]
    # if qid not in number:
    #     return
    
    patch_size = 36
    w_featmap=image.shape[-2] // patch_size    
    h_featmap=image.shape[-1] // patch_size 
    nh = 12
    save_name = ["last", "coarse", "fine"]
    
    plt.figure(figsize=(10, 10))
    im2show = np.copy(image.permute(1,2,0).cpu().numpy())
    plt.subplot(1, 1, 1)
    plt.imshow(im2show, cmap='inferno')
    plt.axis('off')
    # text = str(qid.numpy()) + "__" + str(question) + "__" + str(pred) + "__" + str(answer)
    text = str(qid) + "__" + str(question) + "__" + str(answer)
    output_path = 'attn_output/cls/PathVQA1/'+text
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    plt.savefig(output_path+'/pred__'+str(pred)+".jpg")
    for i in range(len(score)):
        plt.figure(figsize=(10, 10))
        attentions = score[i, :, 1:]
        attentions=attentions.reshape(nh, w_featmap, h_featmap)
        attentions=nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size , mode="nearest")[0].cpu().numpy()
        
        plt.subplot(1, 1, 1)
        plt.imshow(im2show, cmap='inferno', alpha=0.8)
        plt.imshow(np.mean(attentions, 0), cmap='inferno', alpha=0.4)
        plt.axis('off')
        # plt.title(text)
        plt.savefig(output_path+'/'+save_name[i]+".jpg")
    
def validate3(config, model, data_loader, loss_fn, device, idx2ans):
    model.eval()
    val_loss = []

    preds = []
    gts =[]
    qids = []
    preds_top = []
    bar = tqdm(data_loader, leave=False)
    m = 0
    img_id = []
    answer_id = []
    draw_id = [87, 431, 472, 840, 842, 946, 948]
    # draw_id = ["test_0236","test_0400","test_0556"]
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    
    with torch.no_grad():
        for (qid, image, tokens, segment_ids, input_mask, answer, question, ques_target) in bar:
            image, answer, ques_target = image.to(device), answer.to(device), ques_target.to(device)
            tokens, segment_ids, input_mask = tokens.to(device), segment_ids.to(device), input_mask.to(device)
            logits, score = model(image=image, tokens=tokens, segment_ids=segment_ids, input_mask=input_mask,
                                  answer=answer, question=question, val=True)

            loss = loss_fn(logits, answer)
            
            loss_np = loss.detach().cpu().numpy()
            pred = logits.argmax(1).detach()
            gt = answer.detach()

            preds.append(pred)
            gts.append(gt)
            for k in range(len(logits)):
                pred_topk_ids = logits[k].argsort()[-3:].detach().cpu().tolist()
                preds_top.append(pred_topk_ids)
                # cat
                # if pred[k] == gt[k] and "/" not in idx2ans[gt[k]] \
                #     and "/" not in question[k] and gt[k] not in answer_id and idx2ans[gt[k]] not in ["yes", "no"]:
                #     draw_image2(config, image[k], qid[k], question[k], idx2ans[gt[k]], idx2ans[pred[k]], score[k], path="positive")
                #     m += 1
                #     img_id.append(qid[k])
                #     answer_id.append(gt[k])
                # if pred[k] != gt[k] and "/" not in idx2ans[gt[k]] and "/" not in idx2ans[pred[k]] and "/" not in question[k]:
                #     draw_image2(config, image[k], qid[k], question[k], idx2ans[gt[k]], idx2ans[pred[k]], score[k], path="negative")
                #
                # if qid[k] in draw_id:
                #     draw_image2(config, image[k], qid[k], question[k], idx2ans[gt[k]], idx2ans[pred[k]], score[k], path="negative")
                #
                # # split
                # if pred[k] == gt[k] and "/" not in idx2ans[gt[k]] \
                #     and "/" not in question[k] and idx2ans[gt[k]] not in ["yes", "no"]:
                # # if qid[k] in draw_id:
                #     draw_image(config, image[k], qid[k], question[k], idx2ans[gt[k]], idx2ans[pred[k]], score[k])
                #     m += 1
                #     img_id.append(qid[k])
                #     answer_id.append(gt[k])
                
                # BAN
                if qid[k] in draw_id:
                    draw_image3(config, image[k], qid[k], question[k], idx2ans[gt[k]], idx2ans[pred[k]], score[k])
                
                
                
                # word
                if pred[k] == gt[k] and "/" not in idx2ans[gt[k]] and "/" not in question[k]:
                    draw_word(config, image[k], qid[k], question[k], idx2ans[gt[k]], idx2ans[pred[k]], score[k], 
                              path="word", tokens=tokens[k], input_mask=input_mask[k], tokenizer=tokenizer)
                    m += 1
                    img_id.append(qid[k])
                    answer_id.append(gt[k])
            if m > 500:
                break
                
            if not isinstance(qid,list):
                qid = qid.tolist()
            for i in qid:
                qids.append(i)
            val_loss.append(loss_np)

        val_loss = np.mean(val_loss)

    preds = torch.cat(preds).cpu().numpy()
    gts = torch.cat(gts).cpu().numpy()
    
    return val_loss

def main(args, config):
    device = args.device
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    rand_a = int(time.strftime("%Y%m%d%H%M%S")) % 114514
    
    id2ans = json.load(open(os.path.join(config['vqa_root'], config['label2ans'])))
    
    # loading dataset
    print("Creating medical vqa datasets")
    datasets = create_dataset(config=config)
    ans_size = len(id2ans)

    train_loader, val_loader, test_loader = create_loader(datasets,[None, None, None],
                                              collate_fns=[None, None, None],  # 指定collate_fns参数
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4,4,4], is_trains=[True, False, False]) 
    
    print("Creating model")
    
    if args.model == 'vqa3':
        model = BiVision_VQA3(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device)
    elif args.model == 'vqa14':
        model = Bistep_vqa(image_size = config['image_size'], vit = config['vit'], vit_grad_ckpt = config['vit_grad_ckpt'],
                 vit_ckpt_layer = config['vit_ckpt_layer'], output_size = ans_size, device=args.device, config=config)
    elif args.model == 'vqa18':
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
    
    # load_checkpoint
    if args.load_checkpoint:
        pretrained_dict = torch.load(args.load_checkpoint, map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)} #  and k.split(".")[0] not in layers_out)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    # # load_checkpoint
    # if args.load_checkpoint:
    #     pretrained_dict = torch.load(args.load_checkpoint, map_location="cpu")
    #     model_dict = model.state_dict()
    #     # 只保留预训练模型中与当前模型相匹配的部分
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                        k in model_dict and v.shape == model_dict[k].shape}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict, strict=False)

    model.to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    
    test_loss = validate3(config, model, val_loader, loss_fn, device, id2ans)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/VQA_RAD.yaml', type=str)   # PathVQA
    parser.add_argument('--output_dir', default='output/VQA_RAD_test', type=str)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--seed', default=19, type=int) # 26 only sentence bert
    parser.add_argument('--load_checkpoint',
                        # default="ckpt/PathVQA_30_54.pt", # 23
                        # default="ckpt/VQA_RAD_28_74_18.pt",
                        # default="ckpt/VQA_RAD_35_mfb_65.pt",
                        default="ckpt/VQA_RAD_34_ban_71.pt",
                        type=str)
    parser.add_argument('--save_dir', default="ckpt", type=str)
    parser.add_argument('--dataset', default="VQA_RAD", type=str)
    parser.add_argument('--model', default='model', type=str)    # 8
    args = parser.parse_args()

    yaml = yaml.YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)
    
    