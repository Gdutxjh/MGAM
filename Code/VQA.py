import argparse
import os
import numpy as np
import random
import json
import time
import ruamel.yaml as yaml
from pathlib import Path
from tqdm import tqdm
import utils

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from data import create_dataset, create_loader
from data.EVQA_dataset import vqa_collate_fn
from utils import data_eval

from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForQuestionAnswering
from nltk.translate.bleu_score import sentence_bleu
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def train(epoch, tokenizer, model, device, train_loader, optimizer, train_step):
    model.train()
    for i, (image, question_id, question, answer) in enumerate(train_loader):
        # question = question.to(device)
        # answer = answer.to(device)
        
        source = tokenizer.batch_encode_plus(question, max_length= config["question_len"], pad_to_max_length=True, return_tensors='pt')
        target = tokenizer.batch_encode_plus(answer, max_length=  config["answer_len"], pad_to_max_length=True, return_tensors='pt')
        source = source.to(device)
        target = target.to(device)
        
        target_ids = target['input_ids']
        
        y_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone().detach()
        lm_labels[target_ids[:, 1:] == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=source['input_ids'], attention_mask=source['attention_mask'], decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("\r epoch: %d, [%d/%d]" % (epoch, i, train_step), end="")

def evaluate(tokenizer, model, device, loader, val_step):
    """用BLEU4评估"""
    model.eval()
    bleus = []
    acc = 0
    
    result = []
    
    print("max_step: %d" % (val_step))
    with torch.no_grad():
        for i, (image, question_id, question, answer) in tqdm(enumerate(loader, 0), desc='Evaluate'):
            # question = question.to(device)
            # answer = answer.to(device)
            
            source = tokenizer.batch_encode_plus(question, max_length= config["question_len"], pad_to_max_length=True, return_tensors='pt')
            target = tokenizer.batch_encode_plus(answer, max_length=  config["answer_len"], pad_to_max_length=True, return_tensors='pt')

            source = source.to(device)
            target = target.to(device)
            
            target_ids = target['input_ids']
            
            generated_ids = model.generate(
                input_ids = source['input_ids'],
                attention_mask = source['attention_mask'], 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in target_ids]
            bleu_4 = sentence_bleu([tar.split() for tar in target], preds[0].split(), [0, 0, 0, 1])
            bleus.append(bleu_4)
            question_id = question_id.numpy().tolist()
            for j in range(len(preds)):
                if preds[j] == target[j]:
                    acc +=1
                result.append({"question": question[j], "question_id":question_id[j], "answer":preds[j], "target": target[j]})
            # print("\r[%d/%d]" % (i, val_step), end=" ")
    return sum(bleus) / len(bleus), acc/len(result), result
      

def main(args, config):
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset #### 
    print("Creating evqa datasets")
    datasets = create_dataset(config)
    train_size, val_size, test_size = datasets[0].data_size, datasets[1].data_size, datasets[2].data_size
    train_step, val_step, test_step = train_size/config['batch_size_train'], val_size/config['batch_size_test'], test_size/config['batch_size_test']
    train_loader, val_loader, test_loader = create_loader(datasets,[None, None, None],
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[1,1,1], is_trains=[True, False, False], collate_fns = [vqa_collate_fn,None,None]) 
    
    # model
    print("Creating model")
    tokenizer = AutoTokenizer.from_pretrained(config["t5_path"])
    model = T5ForConditionalGeneration.from_pretrained(config["t5_path"])
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    print("Start training")
    start_time = time.time()
    best_bleu = 0
    best_acc = 0
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            train(epoch, tokenizer, model, device, train_loader, optimizer, train_step)
            cur_bleu, acc,result = evaluate(tokenizer, model, device, val_loader, val_step)
            all_acc = data_eval(result, config["csv_path"], args.result_dir, epoch)
            
            if acc > best_acc:
                # torch.save(model.state_dict, os.path.join(config["output_dir"], config["choose"], 't5_best_model.pt'))
                tokenizer.save_pretrained(os.path.join(config["output_dir"], config["choose"], config["save_model"]))
                model.save_pretrained(os.path.join(config["output_dir"], config["choose"], config["save_model"]))
                best_acc = acc
            t = time.localtime(time.time())
            timing = time.asctime(t)
            log = 'time: {},\n epoch: {}, Best acc: {}, Current acc: {}, Current bleu: {}\n'.format(timing, epoch, best_acc, acc, cur_bleu)
            log += all_acc
            print(log)
            with open(os.path.join(args.result_dir, "log.txt"),"a") as f1:
                f1.write(log + "\n")
            with open(os.path.join(args.result_dir, "result_"+str(epoch)+".json"),"w") as f2:
                f2.write(json.dumps(result, indent=2))
                
        else:
            cur_bleu, acc, result = evaluate(tokenizer, model, device, test_loader, test_step)
            print('Current acc: {}, Current bleu: {}'.format(acc, cur_bleu))
            all_acc = data_eval(result, config["csv_path"], args.result_dir, epoch)
            with open(os.path.join(args.result_dir, "result_test.json"),"w") as f2:
                f2.write(json.dumps(result, indent=2))
            return
    cur_bleu, acc, result = evaluate(tokenizer, model, device, test_loader, test_step)
    print('Current acc: {}, Current bleu: {}'.format(acc, cur_bleu))
    all_acc = data_eval(result, config["csv_path"], args.result_dir, epoch)
    with open(os.path.join(args.result_dir, "result_test.json"),"w") as f2:
        f2.write(json.dumps(result, indent=2))
# def predict(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, text: str, device):
#     with torch.no_grad():
#         inputs = tokenizer(text, max_length=MAX_LEN, padding=True, return_tensors='pt')
#         ids = inputs['input_ids']
#         mask = inputs['attention_mask']
#         ids = ids.to(device)
#         mask = mask.to(device)
#         generated_ids = model.generate(
#                 input_ids = ids,
#                 attention_mask = mask, 
#                 max_length=150, 
#                 num_beams=2,
#                 repetition_penalty=2.5, 
#                 length_penalty=1.0, 
#                 early_stopping=True
#                 )
#         preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
#     return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/T5_EVQA.yaml') 
    parser.add_argument('--evaluate', 
                        default=True,
                        action='store_true') 
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args.output_dir = os.path.join(config["output_dir"], config["choose"])
    args.result_dir = os.path.join(args.output_dir, 'result')
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    main(args, config)
    
    
    