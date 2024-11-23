import os 
import json
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import random
import torch.nn as nn

# if __name__ == '__main__':
#     vqa_root = "../../nas/med-data/PathVQA/qas"
#     split = ['train_vqa.pkl', 'val_vqa.pkl', 'test_vqa.pkl']
#     cfile = ["ans2label.pkl", "trainval_ans2label.pkl", "trainval_label2ans.pkl"]
    # train_dataset = pkl.load(open(os.path.join(vqa_root, split[2]), "rb"))
    # print(train_dataset[0])
    # # {'answer_type': 'other', 
    # # 'img_id': 'train_0422', 
    # # 'label': {'in the canals of hering': 1}, 
    # # 'question_id': 100422000, 
    # # 'question_type': 'where', 
    # # 'sent': 'Where are liver stem cells (oval cells) located?'}
    # row = train_dataset[0]
    # print(row['sent'])
    # print('load dataset')

    # ans2label_all = pkl.load(open(os.path.join(vqa_root, cfile[0]), "rb"))
    # print(len(ans2label_all))
    # ans2label = pkl.load(open(os.path.join(vqa_root, cfile[1]), "rb"))
    # print(len(ans2label))


    # label2ans = pkl.load(open(os.path.join(vqa_root, cfile[2]), "rb"))
    # # print(label2ans[[0]])
    # print(len(label2ans))
    # # print('load dataset')


''' BLEU (BiLingual Evaluation Understudy)
@Author: baowj
@Date: 2020/9/16
@Email: bwj_678@qq.com
'''
import numpy as np

class BLEU():
    def __init__(self, n_gram=1):
        super().__init__()
        self.n_gram = n_gram

    def evaluate(self, candidates, references):
        ''' 计算BLEU值
        @param candidates [[str]]: 机器翻译的句子
        @param references [[str]]: 参考的句子
        @param bleu: BLEU值
        '''

        BP = 1
        bleu = np.zeros(len(candidates))
        for k, candidate in enumerate(candidates):
            r, c = 0, 0
            count = np.zeros(self.n_gram)
            count_clip = np.zeros(self.n_gram)
            count_index = np.zeros(self.n_gram)
            p = np.zeros(self.n_gram)
            for j, candidate_sent in enumerate(candidate):
                # 对每个句子遍历
                for i in range(self.n_gram):
                    count_, n_grams = self.extractNgram(candidate_sent, i + 1)
                    count[i] += count_
                    reference_sents = []
                    reference_sents = [reference[j] for reference in references]
                    count_clip_, count_index_ = self.countClip(reference_sents, i + 1, n_grams)
                    count_clip[i] += count_clip_
                    c += len(candidate_sent)
                    r += len(reference_sents[count_index_])
                p = count_clip / count
            rc = r / c
            if rc >= 1:
                BP = np.exp(1 - rc)
            else:
                rc = 1
            p[p == 0] = 1e-100
            p = np.log(p)
            bleu[k] = BP * np.exp(np.average(p))
        return bleu
            

    def extractNgram(self, candidate, n):
        ''' 抽取出n-gram
        @param candidate: [str]: 机器翻译的句子
        @param n int: n-garm值
        @return count int: n-garm个数
        @return n_grams set(): n-grams 
        '''
        count = 0
        n_grams = set()
        if(len(candidate) - n + 1 > 0):
            count += len(candidate) - n + 1
        for i in range(len(candidate) - n + 1):
            n_gram = ' '.join(candidate[i:i+n])
            n_grams.add(n_gram)
        return (count, n_grams)
    
    def countClip(self, references, n, n_gram):
        ''' 计数references中最多有多少n_grams
        @param references [[str]]: 参考译文
        @param n int: n-gram的值s
        @param n_gram set(): n-grams

        @return:
        @count: 出现的次数
        @index: 最多出现次数的句子所在文本的编号
        '''
        max_count = 0
        index = 0
        for j, reference in enumerate(references):
            count = 0
            for i in range(len(reference) - n + 1):
                if(' '.join(reference[i:i+n]) in n_gram):
                    count += 1
            if max_count < count:
                max_count = count
                index = j
        return (max_count, index)


def test_bleu():
    bleu_ = BLEU(4)
    candidates = [['It is a guide to action which ensures that the military always obeys the commands of the party'],
                 ['It is to insure the troops forever hearing the activity guidebook that party direct'],
    ]
    candidates = [[s.split() for s in candidate] for candidate in candidates]
    references = [['It is a guide to action that ensures that the military will forever heed Party commands'],
                  ['It is the guiding principle which guarantees the military forces always being under the command of the Party'],
                  ['It is the practical guide for the army always to heed the directions of the party']
    ]
    references = [[s.split() for s in reference] for reference in references]
    print(bleu_.evaluate(candidates, references))

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sentence_bert():
    # Sentences we want sentence embeddings for
    sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print("Sentence embeddings:")
    print(sentence_embeddings)

def get_tran_spread():
    root = '../med-data/PathVQA/split/trans/train_translate.json'
    output_path = '../med-data/PathVQA/split/trans/train_spread.json'
    data = json.load(open(root, "r"))
    result = []
    closed_ans = ["yes", "no"]
    for i in range(len(data)):
        row = data[i]
        row["sent"] = row["question"]
        answer = row["answer"]
        row["label"] = {str(answer): 1}
        if answer in closed_ans:
            row["answer_type"] = "yes/no"
        else:
            row["answer_type"] = "other"
        result.append(row)
    json.dump(result, open(output_path, "w"), indent=2)
    print("save to %s" % output_path)

def get_top_true():
    root = "output/PathVQA/result/result_test_5_false.json"
    data = json.load(open(root, "r"))
    result = []
    for i in range(len(data)):
        row = data[i]
        if row["gt"] in row ["top_pred"]:
            result.append(row)
    json.dump(result, open("output/result_top.json", "w"), indent=2)

def test_random():
    a = [i for i in range(10)]
    b = random.randint(1, 10)
    c = random.randint(1, 10)
    print(a)
    d = a[b]
    a[b] = a[c]
    a[c] = d
    print(a)

def data_normal_2d(orign_data, dim=-1, device="cpu"):
    """
	针对于3维tensor归一化
	可指定维度进行归一化，默认为行归一化
    """
    d_min = torch.min(orign_data, dim=dim)[0]
    for i in range(d_min.shape[0]):
        for j in range(d_min.shape[1]):
            if d_min[i, j] < 0:
                orign_data[i, j, :] += torch.abs(d_min[i, j]).to(device)
                d_min = torch.min(orign_data, dim=dim)[0]

    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(-1)
        dst = dst.unsqueeze(-1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    return norm_data

def draw():
    attentions = [x*0.1 for x in range(1, 10)]
    attentions = torch.tensor(attentions).reshape(1, 3, 3)
    attentions=nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=36 , mode="nearest")[0].cpu().numpy()

if __name__ == '__main__':
    # sentence_bert()
    # get_tran_spread()
    # get_top_true()
    # test_random()
    
    # root1 = '../med-data/PathVQA/split/trans/train_spread.json'
    # data = json.load(open(root1, "r"))
    # print(len(data))
    # root2 = '../med-data/PathVQA/split/declaration/train_declar.json'
    # data2 = json.load(open(root2, "r"))
    # print(len(data2))
    # data.extend(data2)
    # print(len(data))
    # json.dump(data, open("predata/PathVQA/train.json", "w"), indent=2)
    # print("end")
    # device = "cuda:1"
    # x = torch.randint(low=-10,high=10,size=(3,6,12)).to(device)
    # res = data_normal_2d(x, device=device)
    # print(res)
    load_checkpoint = "ckpt/VQA_RAD_28_74_18.pt"
    pretrained_dict = torch.load(load_checkpoint, map_location='cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)} 
    for k, v in pretrained_dict.items(): 
        print(k)
    # print(pretrained_dict)
    
