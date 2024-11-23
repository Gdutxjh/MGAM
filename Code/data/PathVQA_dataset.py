import os 
import json
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from transformers import BertTokenizer
import copy
import pickle as pkl

from data.text_enhance import tokens_enhance

class PathVQA_datset_1(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        # self.ann = pkl.load(open(os.path.join(config['vqa_root'], config['split'][split]),'rb'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))[0]
        self.ans_size = len(self.ans2label)
        self.ans_embeds = json.load(open(os.path.join(config['vqa_root'], config['ans_embeds'])))[0]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {'answer_type': 'other', 
        # 'img_id': 'train_0422', 
        # 'label': {'in the canals of hering': 1}, 
        # 'question_id': 100422000, 
        # 'question_type': 'where', 
        # 'sent': 'Where are liver stem cells (oval cells) located?'}
        ann = self.ann[index]
        qid = ann['img_id']

        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, ann['img_id']+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        question = ann['question']
        # question = ann['declaration_mask']
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])
        # for key, value in ann['label'].items():
        #     answer = key
        #     weights =  value
        #     # answer = self.proc_ans(answer)
        #     answer_id = self.ans2label[answer]  # the histone subunits
        answer = ann["answer"]
        answer_id = self.ans2label[answer]
        weights = 1
        # if answer_id == 3:
        #     print(qid)
        # ans_embeds = self.ans_embeds[answer]
        answer_type = ann["answer_type"]

        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer_id), answer_type

    def proc_ans(self, ans):
        ans_score = np.zeros(self.ans_size, np.float32)
        ans_score[self.ans2label[ans]] = 1.

        return ans_score

class PathVQA_datset(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []

        # =============注释82，取消81注释
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/')
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        if self.split == "train":
            self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        else:
            self.ann = pkl.load(open(os.path.join(config['vqa_root'], config['split'][split]),'rb'))
        # self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        # =================删除90行最后的[0]
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))
        self.ans_size = len(self.ans2label)
        # self.ans_embeds = json.load(open(os.path.join(config['vqa_root'], config['ans_embeds'])))[0]
        self.closed_id = self.get_closed_ans_id()

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {'answer_type': 'other', 
        # 'img_id': 'train_0422', 
        # 'label': {'in the canals of hering': 1}, 
        # 'question_id': 100422000, 
        # 'question_type': 'where', 
        # 'sent': 'Where are liver stem cells (oval cells) located?'}

        global answer_id
        ann = self.ann[index]
        qid = ann['img_id']

        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, ann['img_id']+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # question = ann['question']
        question = ann['sent']
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])
        for key, value in ann['label'].items():
            answer = key
            weights = value
            # answer = self.proc_ans(answer)
            answer_id = self.ans2label[answer]  # the histone subunits

        # answer_type = ann["answer_type"]
        # answer = ann["answer"]

        # answer_id = self.ans2label[str(answer)]
        # weights = 1
        # if answer_id == 3:
        #     print(qid)
        # ans_embeds = self.ans_embeds[answer]
        answer_type = ann["answer_type"]
        if answer_type == 'CLOSED':
            answer_target = 0
            ques_target = torch.zeros(self.ans_size)
            ques_target[self.closed_id] = 1
        else :
            answer_target = 1
            ques_target = torch.ones(self.ans_size)
            ques_target[self.closed_id] = 0

        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer_id, dtype = torch.long), question, ques_target

    def get_closed_ans_id(self):
        ans_id = []
        answer_id = int(self.ans2label["no"])
        ans_id.append(answer_id)
        answer_id = int(self.ans2label["yes"])
        ans_id.append(answer_id)
        return ans_id

class PathVQA_datset_CR(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        if self.split == "train":
            self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        else:
            self.ann = pkl.load(open(os.path.join(config['vqa_root'], config['split'][split]),'rb'))
        # self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))[0]
        self.ans_size = len(self.ans2label)
        # self.ans_embeds = json.load(open(os.path.join(config['vqa_root'], config['ans_embeds'])))[0]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {'answer_type': 'other', 
        # 'img_id': 'train_0422', 
        # 'label': {'in the canals of hering': 1}, 
        # 'question_id': 100422000, 
        # 'question_type': 'where', 
        # 'sent': 'Where are liver stem cells (oval cells) located?'}
        ann = self.ann[index]
        qid = ann['img_id']

        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, ann['img_id']+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        
        # question = ann['question']
        question = ann['sent']
        tokens, segment_ids, input_mask= (question, self.tokenizer, self.config['seq_max_len'])
        for key, value in ann['label'].items():
            answer = key
            weights =  value
            # answer = self.proc_ans(answer)
            answer_id = self.ans2label[answer]  # the histone subunits
        answer_type = ann["answer_type"]
        if answer_type == 'CLOSED':
            answer_target = 0
            if answer.lower() == "no":
                answer_id = 0
            else:
                answer_id = 1
        else :
            answer_target = 1
        # answer = ann["answer"]
        # answer_id = self.ans2label[str(answer)]
        # weights = 1
        # if answer_id == 3:
        #     print(qid)
        # ans_embeds = self.ans_embeds[answer]
       

        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer_id), question, answer_target

class PathVQA_pretrain(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['pretrian_split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))[0]
        self.ans_size = len(self.ans2label)
        # self.ans_embeds = json.load(open(os.path.join(config['vqa_root'], config['ans_embeds'])))[0]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {'answer_type': 'other', 
        # 'img_id': 'train_0422', 
        # 'label': {'in the canals of hering': 1}, 
        # 'question_id': 100422000, 
        # 'question_type': 'where', 
        # 'sent': 'Where are liver stem cells (oval cells) located?'}
        ann = self.ann[index]
        qid = ann['img_id']

        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, ann['img_id']+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.split == "train":
            question = ann['declaration']
        else:
            question = ann['question']
        
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])
        
        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), \
            torch.tensor(input_mask, dtype = torch.long), question

class VQA_SLAKE_datset(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], config["ans2label"])))
        self.ans_size = len(self.ans2label) # 4946
        self.closed_id = self.get_closed_ans_id()

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {"img_id": 102,
            # "img_name": "xmlab102/source.jpg",
            # "question": "What modality is used to take this image?",
            # "answer": "CT",
            # "q_lang": "en",
            # "location": "Lung",
            # "modality": "CT",
            # "answer_type": "OPEN",
            # "base_type": "vqa",
            # "triple": ["vhead", "_", "_"],
            # "qid": 11934,
            # "content_type": "Modality"}
        ann = self.ann[index]
        qid = int(ann['qid'])

        image_path = os.path.join(self.config['vqa_root'], "imgs", ann['img_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        answer_type = ann["answer_type"]
        # question = ann['declaration_mask']
        question = ann['question']
        answer = ann["answer"]
        answer_id = int(self.ans2label[str(answer)])
        # tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])
        # for key, value in ann['label'].items():
        #     answer = key
        #     weights =  value
        #     # answer = self.proc_ans(answer)
        #     answer_id = self.ans2label[answer]  # the histone subunits
        # answer = ann["answer"]
        # answer_id = int(self.ans2label[str(answer)])
        # weights = 1
        answer_type = ann["answer_type"]
        if answer_type == 'CLOSED':
            answer_target = 0
            ques_target = torch.zeros(self.ans_size)
            ques_target[self.closed_id] = 1
        else :
            answer_target = 1
            ques_target = torch.ones(self.ans_size)
            ques_target[self.closed_id] = 0

        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long),  \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer_id), question, ques_target
    
    def get_closed_ans_id(self):
        ans_id = []
        answer_id = int(self.ans2label["No"])
        ans_id.append(answer_id)
        answer_id = int(self.ans2label["Yes"])
        ans_id.append(answer_id)
        return ans_id
    
class VQA_SLAKE_pretain(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], "en_data", config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], config["ans2label"])))
        self.ans_size = len(self.ans2label) # 4946
        self.mask = {"train": "declaration", "test": "declaration_mask"}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {"img_id": 102,
            # "img_name": "xmlab102/source.jpg",
            # "question": "What modality is used to take this image?",
            # "answer": "CT",
            # "q_lang": "en",
            # "location": "Lung",
            # "modality": "CT",
            # "answer_type": "OPEN",
            # "base_type": "vqa",
            # "triple": ["vhead", "_", "_"],
            # "qid": 11934,
            # "content_type": "Modality"}
        ann = self.ann[index]
        qid = ann['qid']

        image_path = os.path.join(self.config['vqa_root'], "imgs", ann['img_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        answer_type = ann["answer_type"]
        caption = ann[self.mask[self.split]]
        targets = []
        
        tokens_m, segment_ids_m, input_mask_m, targets = \
                get_token_mask_vit(caption, self.tokenizer, self.config['seq_max_len'])
        
        tokens, segment_ids, input_mask = encode_text_vit(caption, self.tokenizer, self.config['seq_max_len'])
        # if answer_type == "OPEN":
        #     tokens, segment_ids, input_mask, targets = \
        #         get_token_mask(caption, self.tokenizer, self.config['seq_max_len'])

        # else:
        #     answer = ann["answer"]
        #     if answer.lower() == "yes":
        #         targets = 0
        #     else:
        #         targets = 1
        
        if self.split == "train":
            return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), \
                torch.tensor(input_mask, dtype = torch.long), \
                torch.tensor(tokens_m, dtype = torch.long), torch.tensor(segment_ids_m, dtype = torch.long),\
                torch.tensor(input_mask_m, dtype = torch.long), \
                torch.tensor(targets, dtype = torch.long), answer_type
        else:
            return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), \
                torch.tensor(input_mask, dtype = torch.long), torch.tensor(targets, dtype = torch.long), answer_type

class VQA_RAD_datset(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/')
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))
        self.ans_size = len(self.ans2label) # 4946
        self.closed_id = self.get_closed_ans_id()
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {"qid": 1, 
        # "image_name": "synpic54610.jpg", 
        # "image_organ": "HEAD", 
        # "answer": "Yes", 
        # "answer_type": "CLOSED", 
        # "question_type": "PRES", 
        # "question": "Are regions of the brain infarcted?", 
        # "phrase_type": "freeform"}
        ann = self.ann[index]
        qid = ann['qid']

        # image_path = os.path.join(self.config['vqa_root'], 'VQA_RAD_Image', ann['image_name'])
        image_path = os.path.join(self.config['vqa_root'], 'image', ann['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        question = ann['question']
        # question = ann['declaration_mask']
        # tokens, segment_ids, input_mask= encode_text_vit(question, self.tokenizer, self.config['seq_max_len'])
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])
        # for key, value in ann['label'].items():
        #     answer = key
        #     weights =  value
        #     # answer = self.proc_ans(answer)
        #     answer_id = self.ans2label[answer]  # the histone subunits
        answer = ann["answer"]
        answer_id = int(self.ans2label[str(answer).lower()])
        weights = 1
        answer_type = ann["answer_type"]
        if answer_type == 'CLOSED':
            answer_target = 0
            ques_target = torch.zeros(self.ans_size)
            ques_target[self.closed_id] = 1
        else :
            answer_target = 1
            ques_target = torch.ones(self.ans_size)
            ques_target[self.closed_id] = 0

        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long),  \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer_id), question, ques_target

    def proc_ans(self, ans):
        ans_score = np.zeros(self.ans_size, np.float32)
        ans_score[self.ans2label[ans]] = 1.

        return ans_score
    
    def get_closed_ans_id(self):
        ans_id = []
        answer_id = int(self.ans2label["no"])
        ans_id.append(answer_id)
        answer_id = int(self.ans2label["yes"])
        ans_id.append(answer_id)
        return ans_id
    
class VQA_RAD_SWIN(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))
        self.ans_size = len(self.ans2label) # 4946
        self.closed_id = self.get_closed_ans_id()
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {"qid": 1, 
        # "image_name": "synpic54610.jpg", 
        # "image_organ": "HEAD", 
        # "answer": "Yes", 
        # "answer_type": "CLOSED", 
        # "question_type": "PRES", 
        # "question": "Are regions of the brain infarcted?", 
        # "phrase_type": "freeform"}
        ann = self.ann[index]
        qid = ann['qid']

        # image_path = os.path.join(self.config['vqa_root'], 'VQA_RAD_Image', ann['image_name'])
        image_path = os.path.join(self.config['vqa_root'], 'image', ann['image_name'])
        image = Image.open(image_path).convert('RGB')
        img = np.array(Image.open(image_path))
        img = (img/255).tolist()
        image = self.transform(image)
        img = self.transform(img)
        
        question = ann['question']
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])
        answer = ann["answer"]
        answer_id = int(self.ans2label[str(answer).lower()])
        answer_type = ann["answer_type"]

        return qid, img, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long),  \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer_id), question

class VQA_RAD_datset_CR(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))
        self.ans_size = len(self.ans2label) # 4946

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {"qid": 1, 
        # "image_name": "synpic54610.jpg", 
        # "image_organ": "HEAD", 
        # "answer": "Yes", 
        # "answer_type": "CLOSED", 
        # "question_type": "PRES", 
        # "question": "Are regions of the brain infarcted?", 
        # "phrase_type": "freeform"}
        ann = self.ann[index]
        qid = ann['qid']

        image_path = os.path.join(self.config['vqa_root'], 'VQA_RAD_Image', ann['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        question = ann['question']
        # question = ann['declaration_mask']
        # tokens, segment_ids, input_mask= encode_text_vit(question, self.tokenizer, self.config['seq_max_len'])
        tokens, segment_ids, input_mask= encode_text_vit(question, self.tokenizer, self.config['seq_max_len'])
        # for key, value in ann['label'].items():
        #     answer = key
        #     weights =  value
        #     # answer = self.proc_ans(answer)
        #     answer_id = self.ans2label[answer]  # the histone subunits
        answer = ann["answer"]
        answer_id = int(self.ans2label[str(answer).lower()])
        weights = 1
        answer_type = ann["answer_type"]
        if answer_type == 'CLOSED':
            answer_target = 0
            if answer.lower() == "no":
                answer_id = 0
            else:
                answer_id = 1
        else :
            answer_target = 1
            

        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long),  \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer_id), question, answer_target

class VQA_RAD_pretrain(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))
        self.ans_size = len(self.ans2label) # 4946
        self.mask = {"train": "declaration", "test": "declaration_mask"}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {"qid": 1, 
        # "image_name": "synpic54610.jpg", 
        # "image_organ": "HEAD", 
        # "answer": "Yes", 
        # "answer_type": "CLOSED", 
        # "question_type": "PRES", 
        # "question": "Are regions of the brain infarcted?", 
        # "phrase_type": "freeform"}
        ann = self.ann[index]
        qid = ann['qid']

        image_path = os.path.join(self.config['vqa_root'], 'VQA_RAD_Image', ann['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        question = ann[self.mask[self.split]]
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])

        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), \
                torch.tensor(input_mask, dtype = torch.long), question

class VQA_RAD_pretrain_lxmert(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))
        self.ans_size = len(self.ans2label) # 4946
        self.mask = {"train": "declaration", "test": "declaration_mask"}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {"qid": 1, 
        # "image_name": "synpic54610.jpg", 
        # "image_organ": "HEAD", 
        # "answer": "Yes", 
        # "answer_type": "CLOSED", 
        # "question_type": "PRES", 
        # "question": "Are regions of the brain infarcted?", 
        # "phrase_type": "freeform"}
        ann = self.ann[index]
        qid = ann['qid']

        image_path = os.path.join(self.config['vqa_root'], 'VQA_RAD_Image', ann['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        question = ann['question']
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.config['seq_max_len'])
        # for key, value in ann['label'].items():
        #     answer = key
        #     weights =  value
        #     # answer = self.proc_ans(answer)
        #     answer_id = self.ans2label[answer]  # the histone subunits
        # answer = ann["answer"]
        # answer_id = int(self.ans2label[str(answer).lower()])
        # weights = 1
        # answer_type = ann["answer_type"]
        # if answer_type == "CLOSED":
        #     if answer.lower() == "yes":
        #         answer_id = 0
        #     else:
        #         answer_id = 1
        answer_type = ann["answer_type"]
        caption = ann[self.mask[self.split]]
        
        if answer_type == "OPEN":
            tokens, segment_ids, input_mask, targets = \
                get_token_mask(caption, self.tokenizer, self.config['seq_max_len'])

        else:
            answer = ann["answer"]
            if answer.lower() == "yes":
                targets = 0
            else:
                targets = 1
            tokens, segment_ids, input_mask= encode_text(caption, self.tokenizer, self.config['seq_max_len'])

        return qid, image, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long),  \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(targets), answer_type

class PathVQA_CPT_datset(Dataset):
    def __init__(self, config, transform, split="train"):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        
        # self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/') 
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(config['CPT_data_root'],'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], 'ans2id.json')))[0]
        self.ans_qid_list = json.load(open(config['ans_qid_list'],'r'))[0]
        self.img_id_list = json.load(open(config['img_id_list'],'r'))

    def __len__(self):
        return self.ann_size

    def __getitem__(self, index):
        # {"question_id": 0, 
        # "img_id": "train_0001", 
        # "question": " What form a submucosal nodule composed of tumor cells embedded in dense fibrous tissue?", 
        # "answer": "carcinoid tumors", 
        # "neg_question": ["Does carcinoid tumor form submucosal nodules embedded by tumor cells in dense fibrous tissue?", "Does carcinoid tumor form submucosal nodules embedded by tumor cells in dense fibrous tissue?", "Do carcinoid tumors form a submucosal node consisting of tumor cells embedded in dense fibrous tissue?", "Does the cytological characteristics of CIS form submucosal nodules, which are composed of tumor cells embedded in dense fibrous tissue?", "Do the cytological features of cis form a submucosal node consisting of tumor cells embedded in dense fibrous tissue?"], 
        # "pos_question": ["How are submucosal nodules formed by tumor cells embedded in dense fibrous tissue?", "What is a submucosal nodule composed of tumor cells embedded in dense fibrous tissue?", "The submucosa is composed of tumor cells embedded with dense fibers. What does it look like?", "What is a submucous nodule, composed of tumor cells wrapped in dense fibrous tissue?", "What form of a submucosal node consisting of tumor cells embedded in dense fibrous tissue?"], 
        # "pos_answer_id": []}
        ann = self.ann[index]
        qid_list = []
        # tokens = []
        # segment_ids = []
        # input_mask = []
        
        qid = ann['question_id']
        img_id = ann['img_id']

        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, img_id+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        question = ann['question']
        tokens, segment_ids, input_mask= vqa_encode_text(question, self.tokenizer, self.config['seq_max_len'])
        answer = ann["answer"]
        answer_id = self.ans2label[answer]
        # weights = torch.Tensor([1, 1, 1])
        
        qid_list.append(qid)
        tokens = torch.tensor(tokens, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        input_mask = torch.tensor(input_mask, dtype = torch.long)
        answer_id = torch.tensor([answer_id])
        
        pos_qid, pos_image, pos_tokens, pos_segment_ids, pos_input_mask, pos_answer_id = self.get_pos_ques(ann)
        qid_list.append(pos_qid)
        image = torch.cat((image.unsqueeze(0), pos_image.unsqueeze(0)), dim=0)
        tokens = torch.cat((tokens.unsqueeze(0), pos_tokens.unsqueeze(0)), dim=0)
        segment_ids = torch.cat((segment_ids.unsqueeze(0), pos_segment_ids.unsqueeze(0)), dim=0)
        input_mask = torch.cat((input_mask.unsqueeze(0), pos_input_mask.unsqueeze(0)), dim=0)
        answer_id = torch.cat((answer_id, pos_answer_id), dim=0)
        
        pos_qid, pos_image, pos_tokens, pos_segment_ids, pos_input_mask, pos_answer_id = self.get_pos_ans(ann)
        qid_list.append(pos_qid)
        image = torch.cat((image, pos_image.unsqueeze(0)), dim=0)
        tokens = torch.cat((tokens, pos_tokens.unsqueeze(0)), dim=0)
        segment_ids = torch.cat((segment_ids, pos_segment_ids.unsqueeze(0)), dim=0)
        input_mask = torch.cat((input_mask, pos_input_mask.unsqueeze(0)), dim=0)
        answer_id = torch.cat((answer_id, pos_answer_id), dim=0)
        
        neg_qid, neg_image, neg_tokens, neg_segment_ids, neg_input_mask, neg_answer_id = self.get_neg_random(ann)
        qid_list.append(neg_qid)
        image = torch.cat((image, neg_image.unsqueeze(0)), dim=0)
        tokens = torch.cat((tokens, neg_tokens.unsqueeze(0)), dim=0)
        segment_ids = torch.cat((segment_ids, neg_segment_ids.unsqueeze(0)), dim=0)
        input_mask = torch.cat((input_mask, neg_input_mask.unsqueeze(0)), dim=0)
        answer_id = torch.cat((answer_id, neg_answer_id), dim=0)
        
        if len(ann["neg_question"]) >= 1:
            neg_qid, neg_image, neg_tokens, neg_segment_ids, neg_input_mask, _ = self.get_neg_ques(ann)
            qid_list.append(neg_qid)
            image = torch.cat((image, neg_image.unsqueeze(0)), dim=0)
            tokens = torch.cat((tokens, neg_tokens.unsqueeze(0)), dim=0)
            segment_ids = torch.cat((segment_ids, neg_segment_ids.unsqueeze(0)), dim=0)
            input_mask = torch.cat((input_mask, neg_input_mask.unsqueeze(0)), dim=0)
            weights = torch.Tensor([1, 1, 1, 2, 2, 2])
            answer_id = torch.cat((answer_id, neg_answer_id), dim=0)
            
        neg_qid, neg_image, neg_tokens, neg_segment_ids, neg_input_mask, _ = self.get_neg_img(ann)
        qid_list.append(neg_qid)
        image = torch.cat((image, neg_image.unsqueeze(0)), dim=0)
        tokens = torch.cat((tokens, neg_tokens.unsqueeze(0)), dim=0)
        segment_ids = torch.cat((segment_ids, neg_segment_ids.unsqueeze(0)), dim=0)
        input_mask = torch.cat((input_mask, neg_input_mask.unsqueeze(0)), dim=0)
        answer_id = torch.cat((answer_id, neg_answer_id), dim=0)

        return torch.tensor(qid_list), image, tokens, segment_ids, input_mask, answer_id
                
    def get_pos_ques(self, ann):
        qid = ann['question_id']
        img_id = ann['img_id']
        
        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, img_id+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        ques_idx = random.randint(0, len(ann["pos_question"])-1)
        ques = ann["pos_question"][ques_idx]
        tokens, segment_ids, input_mask= vqa_encode_text(ques, self.tokenizer, self.config['seq_max_len'])
        
        answer = ann["answer"]
        answer_id = self.ans2label[answer]
        
        tokens = torch.tensor(tokens, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        input_mask = torch.tensor(input_mask, dtype = torch.long)
        answer_id = torch.tensor([answer_id])
        
        return qid, image, tokens, segment_ids, input_mask, answer_id
    
    def get_pos_ans(self, ann):
        ann_qid = ann['question_id']
        idxs = self.ans_qid_list[ann["answer"]]
        idxs.remove(ann_qid)
        if len(idxs) <= 1:
            return self.get_pos_ques(ann)
        
        idx = random.randint(0, len(idxs)-1)
        pos_ann = self.ann[idxs[idx]]
        
        qid = pos_ann['question_id']
        img_id = pos_ann['img_id']
        
        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, img_id+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        ques = pos_ann["question"]
        tokens, segment_ids, input_mask= vqa_encode_text(ques, self.tokenizer, self.config['seq_max_len'])
        
        answer = pos_ann["answer"]
        answer_id = self.ans2label[answer]
        
        tokens = torch.tensor(tokens, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        input_mask = torch.tensor(input_mask, dtype = torch.long)
        answer_id = torch.tensor([answer_id])
        
        return qid, image, tokens, segment_ids, input_mask, answer_id
    
    def get_neg_ques(self, ann):
        qid = ann['question_id']
        img_id = ann['img_id']
        
        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, img_id+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        ques_idx = random.randint(0, len(ann["neg_question"])-1)
        ques = ann["neg_question"][ques_idx]
        tokens, segment_ids, input_mask= vqa_encode_text(ques, self.tokenizer, self.config['seq_max_len'])
        
        answer = ann["answer"]
        answer_id = self.ans2label[answer]
        
        tokens = torch.tensor(tokens, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        input_mask = torch.tensor(input_mask, dtype = torch.long)
        answer_id = torch.tensor([answer_id])
        
        return qid, image, tokens, segment_ids, input_mask, answer_id
    
    def get_neg_img(self, ann):
        qid = ann['question_id']
        img_id = ann['img_id']
        img_id_list = copy.deepcopy(self.img_id_list)
        img_id_list.remove(img_id)
        idx = random.randint(0, len(img_id_list)-1)
        neg_img_id = img_id_list[idx]
        
        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, neg_img_id+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        ques = ann["question"]
        tokens, segment_ids, input_mask= vqa_encode_text(ques, self.tokenizer, self.config['seq_max_len'])
        
        answer = ann["answer"]
        answer_id = self.ans2label[answer]
        
        tokens = torch.tensor(tokens, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        input_mask = torch.tensor(input_mask, dtype = torch.long)
        answer_id = torch.tensor([answer_id])
        
        return qid, image, tokens, segment_ids, input_mask, answer_id
    
    def get_neg_random(self, ann):
        ann_answer = ann["answer"]
        idx = random.randint(0, self.ann_size-1)
        while(ann_answer == self.ann[idx]["answer"]):
            idx = random.randint(0, self.ann_size-1)
        
        neg_ann = self.ann[idx]
        
        qid = neg_ann['question_id']
        img_id = neg_ann['img_id']
        
        image_path = os.path.join(self.config['vqa_root'], 'images', self.split, img_id+".jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        ques = neg_ann["question"]
        tokens, segment_ids, input_mask= vqa_encode_text(ques, self.tokenizer, self.config['seq_max_len'])
        
        answer = neg_ann["answer"]
        answer_id = self.ans2label[answer]
        
        tokens = torch.tensor(tokens, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        input_mask = torch.tensor(input_mask, dtype = torch.long)
        answer_id = torch.tensor([answer_id])
        
        return qid, image, tokens, segment_ids, input_mask, answer_id

# def vqa_encode_text(caption, tokenizer, max_words):
#     #get token ids and remove [CLS] and [SEP] token id
#     part2 = tokenizer.encode(caption)

#     tokens = part2
#     if len(tokens) > max_words:
#         tokens = tokens[:max_words]
#     segment_ids = [1]*(len(tokens))
#     input_mask = [1]*len(tokens)
#     n_pad = max_words - len(tokens)
#     tokens.extend([0]*n_pad)
#     segment_ids.extend([0]*n_pad)
#     input_mask.extend([0]*n_pad)
    
#     return tokens, segment_ids, input_mask  

class bert_datset(Dataset):
    def __init__(self, config):
        self.config = config
        self.ann = []
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.ann = json.load(open(os.path.join(config['vqa_root']),'r'))
        self.data_size = len(self.ann)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        sent = ann['caption']

        # tokens, segment_ids, input_mask, target= get_mlm_mask(self.config, sent, self.tokenizer, self.config['seq_max_len'])
        tokens, segment_ids, input_mask, target= get_token_mask(sent, self.tokenizer, self.config['seq_max_len'])
        return torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long),  \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(target, dtype = torch.long)

def encode_text_vit(caption,tokenizer, max_words):
    part1 = [0 for _ in range(37)]
    #get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)[1:-1]

    tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + part2[:max_words-40] + [tokenizer.sep_token_id]
    if len(tokens) > max_words:
        tokens = tokens[:max_words]
    segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:max_words-40])+1)
    if len(segment_ids) > max_words:
        segment_ids = segment_ids[:max_words]
    input_mask = [1]*len(tokens)
    n_pad = max_words - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)
    
    return tokens, segment_ids, input_mask 

def encode_text(caption, tokenizer, max_words):
    #get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)

    tokens = part2
    # tokens = tokens_enhance(tokens)
    if len(tokens) > max_words:
        tokens = tokens[:max_words]
    segment_ids = [1]*(len(tokens))
    input_mask = [1]*len(tokens)
    n_pad = max_words - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)
    
    return tokens, segment_ids, input_mask 

def encode_text_13(caption, tokenizer, max_words):
    #get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)

    tokens = part2 # [tokenizer.cls_token_id] + [tokenizer.sep_token_id] + part2 + [tokenizer.sep_token_id]
    if len(tokens) > max_words:
        tokens = tokens[:max_words]
    segment_ids = [1]*(len(tokens))
    input_mask = [1]*len(tokens)
    n_pad = max_words - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)
    
    return tokens, segment_ids, input_mask 

class ROCO_dataset(Dataset):
    def __init__(self, split, tfm, args, mode = 'train'):
        self.args = args
        self.data = json.load(open(os.path.join(args['vqa_root'], args["split"][split]), "r"))
        self.tfm = tfm
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        self.mode = mode
        self.keywords = json.load(open(os.path.join(args['vqa_root'], args['keywords']), "r"))
        self.data_size = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # "id": "ROCO_00020",
        # "name": "PMC3970251_CRIONM2014-931546.003.jpg",
        # "caption": " Axial computed tomography scan of the pelvis showing a diffuse infiltration of the bladder wall, catheter in situ (arrow)."
        ann = self.data[idx]
        name = ann["name"]
        caption = ann["caption"].strip()
        split = ann["split"]
        
        path = os.path.join(self.args['vqa_root'], split, 'radiology', 'images', name)
        img = Image.open(path).convert('RGB')


        if self.tfm:
            img = self.tfm(img)
            
        # tokens, segment_ids, input_mask, targets = pretrain_encode_text(caption, self.tokenizer, self.keywords, self.args)
        tokens, segment_ids, input_mask, targets = get_token(self.args, caption, self.tokenizer, self.args['max_position_embeddings'], keywords)

        return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), \
            torch.tensor(input_mask, dtype = torch.long), torch.tensor(targets, dtype = torch.long)

def random_word(config, tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = config['word_mask_rate']
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(0)

    return tokens, output_label

def get_mlm_mask(config, caption, tokenizer, max_seq_length):
    tokens = tokenizer.tokenize(caption)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    
    masked_tokens, masked_label = random_word(config, tokens, tokenizer)
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    lm_label_ids = ([0] + masked_label + [0])
    # Mask & Segment Word
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    return input_ids, segment_ids, input_mask, lm_label_ids

def get_token(config, caption, tokenizer, max_seq_length):
    
    tokens = tokenizer.tokenize(caption)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    masked_tokens, masked_label = random_word(config, tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    # masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    lm_label_ids = ([0] + masked_label + [0])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    return input_ids, segment_ids, input_mask, lm_label_ids

def get_token_mask(caption, tokenizer, max_seq_length):
    tokens = tokenizer.tokenize(caption)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    
    # masked_tokens, masked_label = random_word(config, tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    # masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    idx = random.randint(0,len(input_ids)-1)
    lm_label_ids = input_ids[idx]
    
    tokens[idx] = "[MASK]"
    masked_tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # lm_label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    # assert len(lm_label_ids) == max_seq_length
    return input_ids, segment_ids, input_mask, lm_label_ids

def get_token_mask_vit(caption, tokenizer, max_seq_length):
    part1 = [0 for _ in range(37)]
    tokens = tokenizer.tokenize(caption)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    
    # masked_tokens, masked_label = random_word(config, tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    # masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    idx = random.randint(0,len(input_ids)-1)
    lm_label_ids = input_ids[idx]
    
    tokens[idx] = "[MASK]"
    masked_tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + tokens + [tokenizer.sep_token_id]
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
    # Mask & Segment Word
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # lm_label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    # assert len(lm_label_ids) == max_seq_length
    return input_ids, segment_ids, input_mask, lm_label_ids

def encode_text_5(caption, tokenizer, max_words):
    part1 = [0 for _ in range(5)]
    #get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)[1:-1]

    tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + part2[:max_words-8] + [tokenizer.sep_token_id]
    segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:max_words-8])+1)
    input_mask = [1]*len(tokens)
    n_pad = max_words - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)
    
    return tokens, segment_ids, input_mask  

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list = [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list.append(answer)
        weight_list.append(weights)      
    return torch.stack(image_list, dim=0), torch.Tensor(question_list), torch.Tensor(answer_list), torch.Tensor(weight_list)     
    
        