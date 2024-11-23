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

class SLAKE_datset(Dataset):
    def __init__(self, config, transform, split):
        self.config = config
        self.transform = transform
        self.split = split
        self.ann = []
        self.k = config["topk"]
        
        self.ann = json.load(open(os.path.join(config['vqa_root'], config['split'][split]),'r'))
        self.ann_size = len(self.ann)
        self.ans2label = json.load(open(os.path.join(config['vqa_root'], config["ans2label"])))[0]
        self.ans_size = len(self.ans2label) # 4946
        
        self.json_set_path = config["json_set_path"]
        self.json_set = self.get_json_set()
        
        # {"xmlab132": {
        # "modality": "X-Ray",
        # "top_score": [
        #   [
        #     325,
        #     0.4502550959587097
        #   ],]}
        self.retrieval = json.load(open(config["retrieval_path"]))

    def get_json_set(self):
        choose = ["CT", "MRI", "X-Ray"]
        json_set = {}
        # {
        #     "image": "PMC535567_F2_845.jpg",
        #     "caption": "A \u2013 Chest radiograph of a patient with Pneumocystis jiroveci pneumonia showing bilateral, diffuse interstitial infiltrates B \u2013 Contrast enhanced computed tomographic (CT) scan of chest showing mediastinal lymphadenopathy in a patient with disseminated tuberculosis. Typical central necrosis evident as low attenuation areas (arrows) is seen C \u2013 Contrast enhanced CT scan of brain showing ring enhancing lesions in the basal ganglia bilaterally (arrows). Serology was positive for toxoplasma infection D \u2013 Ophthalmoscopic image of a patient with cytomegalovirus retinitis E \u2013 Non-Hodgkin's lymphoma in a HIV-infected lady presenting as unilateral maxillary swelling F \u2013 Contrast enhanced CT scan of abdomen reveals an oedematous and enlarged pancreas (asterisk) suggestive of acute pancreatitis. The patient was on didanosine and improved following withdrawal of the same and supportive treatment.",
        #     "pmcid": "PMC535567",
        #     "url_name": "1471-2334-4-52-2.jpg"
        #   },
        for i in range(len(choose)):
            json_path = os.path.join(self.json_set_path, "pmc_radio_"+choose[i]+".json")
            retrieval = json.load(open(json_path))
            json_set[choose[i]] = retrieval
        return json_set
    
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # {"img_id": 0, 
        # "img_name": "xmlab0/source.jpg", 
        # "question": "What modality is used to take this image?", 
        # "answer": "MRI", 
        # "q_lang": "en", 
        # "location": "Abdomen", 
        # "modality": "MRI", 
        # "answer_type": "OPEN", 
        # "base_type": "vqa", 
        # "content_type": "Modality", 
        # "triple": ["vhead", "_", "_"], 
        # "qid": 9835}
        ann = self.ann[index]
        qid = ann['qid']
        question = ann['question']

        image_path = os.path.join(self.config['vqa_root'], "imgs", ann['img_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        
        answer = ann["answer"]
        answer_type = ann["answer_type"]
        answer_id = int(self.ans2label[str(answer)])
        
        modality = ann["modality"]
        
        image_name = ann['img_name'].split("/")[0]
        retrieval = self.retrieval[image_name]["top_score"]
        retrieval_text = []
        for i in range(self.k):
            text = self.json_set[modality][retrieval[i][0]]["caption"]
            retrieval_text.append(text)
        
        return qid, image, question, answer, answer_type, torch.tensor(answer_id), retrieval_text
    
    
    
def slake_collate_fn(batch):
    qid_list, image_list, question_list, answer_list, answer_type_list, answer_id_list, retrieval_text_list = [], [], [], [], [], [], []
    for qid, image, question, answer, answer_type, answer_id, retrieval_text in batch:
        qid_list.append(qid)
        image_list.append(image)
        question_list.append(question)      
        answer_list.append(answer)
        answer_type_list.append(answer_type)
        answer_id_list.append(answer_id)
        retrieval_text_list+=retrieval_text
    return qid_list, torch.stack(image_list,dim=0), question_list, answer_list, \
        answer_type_list, torch.stack(answer_id_list, dim=0), retrieval_text_list
