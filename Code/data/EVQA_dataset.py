import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from transformers import AutoTokenizer

class evqa_dataset(Dataset):
    def __init__(self, transform, config, split="train"):
        self.transform = transform
        self.config = config
        
        self.topk = config["top_k"]
        self.annotation = []
        if split =="train":
            self.annotation = json.load(open(os.path.join(config["anno_root"],config["choose"], 'vqa_train.json'),'r'))
        elif split =="val":
            self.annotation = json.load(open(os.path.join(config["anno_root"],config["choose"], 'vqa_val.json'),'r'))
        else:
            self.annotation = json.load(open(os.path.join(config["anno_root"],config["choose"], 'vqa_test.json'),'r'))
        
        self.retrieval = json.load(open(config["retrieval"], "r"))
        self.data_size = len(self.annotation)
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index): 
        ann = self.annotation[index]
        question_id = ann["question_id"]
        # image_path = os.path.join(self.config["vqa_root"],ann['image']) 
        # image = Image.open(image_path).convert('RGB')   
        # image = self.transform(image)
        image_id = ann["image"]
        
        outside_retrieval = "Prompt: If it is a simple question, please answer yes or no.  Outside knowledge: "
        if str(question_id) in self.retrieval:
            retrieval = self.retrieval[str(question_id)]
            i = 0
            for t in retrieval:
                if i > self.topk:
                    break
                if len(t)>200:
                    t = t[:200]
                i+=1
                outside_retrieval += t 
       
        question = pre_question(ann['question'])  
        questoin = "question: "+question+outside_retrieval  
        answer = "answer: "+ str(ann["answer"][0])

        return image_id, question_id, questoin, answer
                
        
def vqa_collate_fn(batch):
    image_list, question_id_list, question_list, answer_list = [], [], [], []
    for image_id, question_id, question, answer in batch:
        image_list.append(image_id)
        question_id_list.append(question_id)
        question_list.append(question)
        answer_list.append(answer)
    return image_list, question_id_list, question_list, answer_list
          