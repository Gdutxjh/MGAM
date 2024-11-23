import math
import numpy as np
import time
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_bleu_score(preds,targets, idx2ans):
       
    bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split(),weights=[1]) for pred,target in zip(preds,targets)])
        
    return np.mean(bleu_per_answer)

def data_eval(result, csv_path, output_path, name):
    data = pd.read_csv(csv_path)
    qoestion_id = data["question_id"]
    acc = 0
    acc_dict = {"yes/no": 0, "number": 0, "other": 0}
    sum_dict = {"yes/no": 0, "number": 0, "other": 0}
    ques_is_list = []
    for qid in qoestion_id:
        ques_is_list.append(qid)

    for i in range(len(result)):
        qid = result[i]["question_id"]
        idx = ques_is_list.index(qid)
        ans = result[i]["answer"]
        ans_type = data["answer_type"][idx]
        ground_true = data["answer"][idx]
        sum_dict[ans_type] += 1
        if ground_true == ans:
            acc_dict[ans_type] += 1

    for key, value in acc_dict.items():
        acc += value

    out = "acc: "+str(acc/len(result)*100)+"\n"+\
          "yes/no: "+str(acc_dict["yes/no"]/(sum_dict["yes/no"]+1e-5)*100)+"\n"+\
          "number: "+str(acc_dict["number"]/(sum_dict["number"]+1e-5)*100)+"\n"+\
          "other: "+str(acc_dict["other"]/(sum_dict["other"]+1e-5)*100)
    print(out)
    f = open(output_path+"/result_"+str(name)+".txt", "w", encoding="utf-8")
    f.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+"\n"+out)
    f.close()
    
    return out



