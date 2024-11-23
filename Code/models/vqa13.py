import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertTokenizer, BertModel
# from torchvision import models as torch_models
import math
import numpy as np
import copy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import os

from models.vit import VisionTransformer, interpolate_pos_embed
from models.resnet import resnet152 
from models.med import BertConfig, BertLMHeadModel
# from models.CoAttn import MFB, CoAtt
# # from models.gumbel_softmax import gumbel_softmax
# from models.lxmert_model import LXRTFeatureExtraction as VisualBertForLXRFeature
# from models.lxmert_model import LXRTModel, LXRTEncoder
# from functools import partial
from models.vqa3 import create_vit, BertLayer
from models.gumbel_softmax import gumbel_softmax

class BiVision_VQA13(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0",
                 top_k = 10,
                 temperature = 1,
                 config = None
                 ):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.device = device
        self.encoder_config = BertConfig.from_json_file(med_config)
        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.sentence_bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel(config=modelConfig) 
        # self.sentence_bert = SentenceTransformer('all-distilroberta-v1')
        
        self.attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        
        # self.local_encoder = LXRTEncoder(modelConfig)
        self.open_transformers = Transformer(self.encoder_config)
        self.close_transformers = Transformer(self.encoder_config)
        
        self.encoder_config.encoder_width = vision_width
    
        self.open_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.close_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        
        # self.classifier = nn.Sequential(nn.Linear(1024, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        self.open_classifier = nn.Sequential(nn.Linear(768, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        self.close_classifier = nn.Sequential(nn.Linear(768, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, 2))
        
        self.loss_triplet= nn.TripletMarginLoss(margin=1.0, p=2)
        self.loss_ce = nn.CrossEntropyLoss()
        
        self.config = config
        ans_embeds_all = torch.load(config["answer_embeds_path"])
        self.ans_embeds_all = ans_embeds_all.to(device)

    def forward(self, image, tokens, segment_ids, input_mask, answer, id2ans, ans_tokens_list, \
                question=None, answer_target=None, qid=None, train=False, not_pretrain=True, val=False):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768
        
        # image_global_embeds = self.resnet(image)  # batch_size*768

        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        sentence_embeds = self.sentence_bert.encode(question)
        sentence_embeds = torch.tensor(sentence_embeds).to(self.device)
        question_embeds = self.attention(question_embeds, sentence_embeds.unsqueeze(1), sentence_embeds.unsqueeze(1))
        
        # get open & close feature
        v_open, v_close, q_open, q_close, mask_open, mask_close, a_open, a_close, qid_open, qid_close = \
            seperate(image_local_embeds, question_embeds, answer, input_mask, answer_target, qid)
        
        logits_open = None
        if len(a_open) > 0:
            h_open, attn_score_open = self.open_transformers(v_open, q_open, mask_open)
            logits_open = self.open_classifier(h_open.mean(0)[:, 0, :])
        
        logits_close = None
        if len(a_close) > 0:
            h_close, attn_score_close = self.close_transformers(v_close, q_close, mask_close)
            logits_close = self.close_classifier(h_close.mean(0)[:, 0, :])

        
        return logits_open, logits_close, a_open, a_close, qid_open, qid_close
    
    def contrastive_loss(self, x, x_aug, similarity_matrix, label):
        # batch_size, _ = x.size()
        # x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=2).squeeze()

        # sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        # sim_matrix = similarity_matrix / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = similarity_matrix / x_aug_abs
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        # loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # set pos_sim as 0
        ori_matrix = torch.tensor(copy.deepcopy(sim_matrix.tolist())).to(self.device)
        sim_matrix[label.bool()] = 0
        denom = sim_matrix.sum(dim=1)
        loss = []
        for i in range(len(label)):
            for j in range(len(label[i])):
                if label[i][j] == 1:
                    loss_i =  ori_matrix[i][j] / denom[i]
                    loss.append(loss_i)
        # loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(torch.tensor(loss)).mean()
        return loss
    
    def info_nce_loss(self, similarity_matrix, labels):

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([negatives], dim=1)

        logits = logits / self.T
        return logits
    
    def get_top_ans_embeds(self, logits, answer, id2ans, ans_tokens_list):
        preds_top = []
        for k in range(len(logits)):
            pred_topk_ids = logits[k].argsort()[-self.top_k:].detach().cpu().tolist()
            preds_top.append(pred_topk_ids)
        
        tokens_list = []
        segment_ids_list = []
        input_mask_list = []
        target = []
        for i in range(len(preds_top)):
            for j in range(len(preds_top[i])):
                ans_input = ans_tokens_list[id2ans[preds_top[i][j]]]
                ans_tokens = ans_input["tokens"]
                ans_segment_ids = ans_input["segment_ids"]
                ans_input_mask = ans_input["input_mask"]
                tokens_list.append(ans_tokens)
                segment_ids_list.append(ans_segment_ids)
                input_mask_list.append(ans_input_mask)
                if preds_top[i][j] == answer[i]:
                    target.append(1)
                else:
                    target.append(0)
            ans_input = ans_tokens_list[id2ans[answer[i]]]
            ans_tokens = ans_input["tokens"]
            ans_segment_ids = ans_input["segment_ids"]
            ans_input_mask = ans_input["input_mask"]
            tokens_list.append(ans_tokens)
            segment_ids_list.append(ans_segment_ids)
            input_mask_list.append(ans_input_mask)
            target.append(1)
                
        return torch.tensor(tokens_list, dtype=torch.long), torch.tensor(segment_ids_list, dtype=torch.long), \
            torch.tensor(input_mask_list, dtype=torch.long), torch.tensor(target, dtype=torch.long), preds_top, logits

    def save_ans_embeds(self):
        torch.save(self.ans_embeds_all, self.config["answer_embeds_path"])

# def multi_classifer_loss(preds, target):
#     C = preds.shape[0]
#     multi_loss = 0
#     label_ones = torch.tensor(preds.shape[1])
    
#     for i in range(len(C)):
#         multi_loss += (target[i] * torch.log2(label_ones / (label_ones + torch.exp(-preds[i]))) +\
#             (label_ones - target[i]) * torch.log2(torch.exp(-preds[i]) / (label_ones + torch.exp(-preds[i]))))
#     return -1 / C * multi_loss

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers
        
    def forward(self, image_embeds, h, mask):
        h[:,1:38,:] += image_embeds
        # emb_cls = h[:, 0, :]
        # question_embeds = h[:, 1:, :]
        # h = torch.cat((emb_cls.unsqueeze(1), image_embeds, question_embeds), dim=1)
        
        hidden_states = []
        all_attn_scores = []
        for i in range(self.n_layers):
            h, attn_scores = self.blocks(h, mask, i)
            hidden_states.append(h)
            all_attn_scores.append(attn_scores)

        return torch.stack(hidden_states, 0), torch.stack(all_attn_scores, 0)

class Bistep_vqa(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:1",
                 top_k = 10,
                 temperature = 1,
                 config = None
                 ):
        super().__init__()
        self.device = device
        self.top_k = top_k
        self.temperature = temperature
        self.encoder_config = BertConfig.from_json_file(med_config)
        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.sentence_bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)
        self.gru = nn.GRU(input_size=self.encoder_config.hidden_size, hidden_size=self.encoder_config.hidden_size, num_layers=2)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel(config=modelConfig)
        
        self.transfor1 = Encoder(self.encoder_config.hidden_size, self.encoder_config.n_layers, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        self.transfor2 = Encoder(self.encoder_config.hidden_size, self.encoder_config.n_layers, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        self.transfor3 = Text_base_attention2(self.encoder_config.hidden_size, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)

        self.norm_1 = nn.LayerNorm(self.encoder_config.hidden_size)
        
        
        self.classifier1 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.classifier2 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.classifier3 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.classifier_open_close = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_triplet= nn.TripletMarginLoss(margin=1.0, p=2)
        self.multi_label_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
    
    def forward(self, image, tokens, segment_ids, input_mask, answer=None, 
                id2ans=None, ans_tokens_list=None, question=None, ques_target=None, val=None):
        v_feat = self.vit_encoder(image)
        v_mask = torch.ones(v_feat.shape[0], v_feat.shape[1]).to(self.device)
        w_feat = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        s_feat = torch.tensor(self.sentence_bert.encode(question, device=self.device)).to(self.device)
        # s_feat, s_hn = self.gru(w_feat)
        
        # outputs1, v_outputs, s_outputs = self.transfor1(w_feat, v_feat, s_feat.unsqueeze(1))    # [16, 37, 768]
        # outputs2, w_outputs, s_outputs = self.transfor2(outputs1, w_outputs, s_outputs)
        outputs1, x_ws, x_v1, x_v2, score = self.transfor3(v_feat, w_feat, s_feat.unsqueeze(1))
        outputs1 = outputs1[:, 0, :]
        
        ques_logits = self.classifier_open_close(x_ws.mean(1))
        loss_multi = self.multi_label_loss(ques_logits, ques_target)
        
        logits1 = self.classifier1(outputs1)
        logits2 = self.classifier2(x_v1[:, 0, :])
        logits3 = self.classifier3(x_v2[:, 0, :])
        loss_ce1 = self.loss_ce(logits1, answer)
        loss_ce2 = self.loss_ce(logits2, answer)
        loss_ce3 = self.loss_ce(logits3, answer)
        
        logits4 = torch.mul(logits1, ques_logits)
        loss_ce4 = self.loss_ce(logits4, answer)
        
        loss_ce = loss_ce1 + loss_ce2 + loss_ce3 + loss_ce4 + loss_multi
        
        # ques_logits = self.classifier3(w_feat[:, 0, :])
        # loss_ce3 = self.loss_ce(ques_logits, answer)
        
        # tokens_list, segment_ids_list, input_mask_list, target, preds_top, logits3 = self.get_top_ans_embeds(logits1, answer, id2ans, ans_tokens_list)
        # tokens_list, segment_ids_list, input_mask_list, target = \
        #     tokens_list.to(self.device), segment_ids_list.to(self.device), input_mask_list.to(self.device), target.to(self.device)
        # answer_embeds = self.text_encoder(input_ids=tokens_list, token_type_ids=segment_ids_list, position_ids=input_mask_list)[0]
        
        # answer_embeds = F.normalize(answer_embeds[:, 0, :], dim=1)
        # answer_embeds = answer_embeds.view(outputs1.shape[0], -1, outputs1.shape[1]) # [bs, 11, 768]
        # target = target.view(outputs1.shape[0], -1)
        # similarity_matrix = torch.matmul(outputs1.unsqueeze(1), answer_embeds.transpose(1, 2)).squeeze(1)  # [bs, 1, 768] * [bs, 768, 6] -> [bs, 1, 6] -> [bs, 6]
        
        # negatives = get_negative_pair(answer_embeds, similarity_matrix, target)
        # loss_triplet = self.loss_triplet(outputs1, answer_embeds[:, -1, :], negatives)
        
        # outputs2, _, _ = self.transfor2(outputs1.unsqueeze(1), s_outputs, answer_embeds[:, :-1, :])
        # logits2 = self.classifier2(outputs2.squeeze(1))
        # loss_ce2 = self.loss_ce(logits2, answer)
        # loss_ce = loss_ce1 + loss_ce2
        # if val:
        #     return logits1
        
        # return logits2, loss_ce, loss_triplet
        if val:
            return logits1, score
        return logits1, loss_ce
    
    def get_top_ans_embeds(self, logits, answer, id2ans, ans_tokens_list):
        preds_top = []
        for k in range(len(logits)):
            pred_topk_ids = logits[k].argsort()[-self.top_k:].detach().cpu().tolist()
            preds_top.append(pred_topk_ids)
        
        tokens_list = []
        segment_ids_list = []
        input_mask_list = []
        target = []
        for i in range(len(preds_top)):
            for j in range(len(preds_top[i])):
                ans_input = ans_tokens_list[id2ans[preds_top[i][j]]]
                ans_tokens = ans_input["tokens"]
                ans_segment_ids = ans_input["segment_ids"]
                ans_input_mask = ans_input["input_mask"]
                tokens_list.append(ans_tokens)
                segment_ids_list.append(ans_segment_ids)
                input_mask_list.append(ans_input_mask)
                if preds_top[i][j] == answer[i]:
                    target.append(1)
                else:
                    target.append(0)
            ans_input = ans_tokens_list[id2ans[answer[i]]]
            ans_tokens = ans_input["tokens"]
            ans_segment_ids = ans_input["segment_ids"]
            ans_input_mask = ans_input["input_mask"]
            tokens_list.append(ans_tokens)
            segment_ids_list.append(ans_segment_ids)
            input_mask_list.append(ans_input_mask)
            target.append(1)
                
        return torch.tensor(tokens_list, dtype=torch.long), torch.tensor(segment_ids_list, dtype=torch.long), \
            torch.tensor(input_mask_list, dtype=torch.long), torch.tensor(target, dtype=torch.long), preds_top, logits

class Bistep_vqa2(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:1",
                 top_k = 10,
                 temperature = 1,
                 config = None
                 ):
        super().__init__()
        self.device = device
        self.top_k = top_k
        self.temperature = temperature
        self.encoder_config = BertConfig.from_json_file(med_config)
        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.sentence_bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)
        self.sentence_bert = SentenceTransformer('./pre_model/mpnet-base', device=self.device)
        # self.gru = nn.GRU(input_size=self.encoder_config.hidden_size, hidden_size=self.encoder_config.hidden_size, num_layers=2)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel(config=modelConfig)
        
        self.transfor1 = Encoder(self.encoder_config.hidden_size, self.encoder_config.n_layers, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        self.transfor2 = Encoder(self.encoder_config.hidden_size, self.encoder_config.n_layers, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        self.transfor3 = Text_base_attention5(self.encoder_config.hidden_size, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        # self.transfor3 = Transfors(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)

        self.norm_1 = nn.LayerNorm(self.encoder_config.hidden_size)
        
        self.ff1 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, self.encoder_config.hidden_size))
        self.ff2 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, self.encoder_config.hidden_size))
        self.ff3 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, self.encoder_config.hidden_size))
        
        self.classifier1 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.classifier2 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.classifier3 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        self.classifier4 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.classifier_open_close = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_triplet= nn.TripletMarginLoss(margin=1.0, p=2)
        self.multi_label_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
        
        # self.config = config
        # ans_embeds_all = torch.load(os.path.join(config["vqa_root"], config["answer_embeds_path"]))
        # self.ans_embeds_all = ans_embeds_all.to(self.device)
    
    def forward(self, image, tokens, segment_ids, input_mask, answer=None, 
                id2ans=None, ans_tokens_list=None, question=None, ques_target=None, val=None):
        v_feat = self.vit_encoder(image)
        # v_mask = torch.ones(v_feat.shape[0], v_feat.shape[1]).to(self.device)
        w_feat = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        s_feat = torch.tensor(self.sentence_bert.encode(question, device=self.device)).to(self.device)
        # w_feat, s_hn = self.gru(w_feat)
        
        # s_feat = w_feat[:, 0, :]
        # w_feat = w_feat[:, 1:, :]
        
        # outputs1, v_outputs, s_outputs = self.transfor1(w_feat, v_feat, s_feat.unsqueeze(1))    # [16, 37, 768]
        # outputs2, w_outputs, s_outputs = self.transfor2(outputs1, w_outputs, s_outputs)
        outputs1, x_ws, x_sv, x_wv = self.transfor3(v_feat, w_feat, s_feat.unsqueeze(1))
        outputs1 = outputs1.mean(1)
        # logits4 = self.classifier4(outputs1)
        # outputs1 = self.ff1(outputs1)
        
        # ques_logits = self.classifier_open_close(x_ws.mean(1))
        # loss_multi = self.multi_label_loss(ques_logits, ques_target)
        
        logits1 = self.classifier1(outputs1)
        # loss_ce4 = self.loss_ce(logits1, answer)
        # logits1 = torch.matmul(outputs1.unsqueeze(1), self.ans_embeds_all.transpose(0, 1)).squeeze(1)
        # logits1 = torch.mul(logits1, ques_logits)
        
        # classifier
        logits2 = self.classifier2(x_sv[:, 0, :])
        logits3 = self.classifier3(x_wv[:, 0, :])
        
        
        # embedding
        # outputs2 = self.ff2(x_v1[:, 0, :])
        # outputs3 = self.ff3(x_v2[:, 0, :])
        # logits2 = torch.matmul(outputs2.unsqueeze(1), self.ans_embeds_all.transpose(0, 1)).squeeze(1)
        # logits3 = torch.matmul(outputs3.unsqueeze(1), self.ans_embeds_all.transpose(0, 1)).squeeze(1)
        # logits2 = torch.mul(logits2, ques_logits)
        # logits3 = torch.mul(logits3, ques_logits)
        
        
        # loss ce
        loss_ce1 = self.loss_ce(logits1, answer)
        loss_ce2 = self.loss_ce(logits2, answer)
        loss_ce3 = self.loss_ce(logits3, answer)
        
        # logits4 = torch.mul(logits1, ques_logits)
        # loss_ce4 = self.loss_ce(logits4, answer)
        
        
        # loss triplet
        # answer_embeds1, target1, preds_top = self.get_top_ans_embeds(logits1, answer, id2ans)
        # similarity_matrix1 = torch.matmul(outputs1.unsqueeze(1), answer_embeds1.transpose(1, 2)).squeeze(1)
        # negatives1 = self.get_negative_pair(answer_embeds1, similarity_matrix1, target1)
        # loss_triplet1 = self.loss_triplet(outputs1, answer_embeds1[:, -1, :], negatives1)
        
        # # loss triplet 2
        # answer_embeds2, target2, _ = self.get_top_ans_embeds(logits2, answer, id2ans)
        # similarity_matrix2 = torch.matmul(outputs2.unsqueeze(1), answer_embeds2.transpose(1, 2)).squeeze(1)
        # negatives2 = self.get_negative_pair(answer_embeds2, similarity_matrix2, target2)
        # loss_triplet2 = self.loss_triplet(outputs2, answer_embeds2[:, -1, :], negatives2)
        
        # answer_embeds3, target3, _ = self.get_top_ans_embeds(logits3, answer, id2ans)
        # similarity_matrix3 = torch.matmul(outputs3.unsqueeze(1), answer_embeds3.transpose(1, 2)).squeeze(1)
        # negatives3 = self.get_negative_pair(answer_embeds3, similarity_matrix3, target3)
        # loss_triplet3 = self.loss_triplet(outputs3, answer_embeds3[:, -1, :], negatives3)
        
        
        # loss = loss_ce1 + loss_ce2 + loss_ce3 + loss_triplet1 + loss_multi + loss_triplet2 + loss_triplet3
        loss =  loss_ce1 + loss_ce2 + loss_ce3
        
        if val:
            return logits1
        
        # if val:
        #     return logits1
        return logits1, loss
    
    def get_top_ans_embeds(self, logits, answer, id2ans):
        preds_top = []
        target = []
        ans_embeds = torch.zeros(logits.shape[0], self.top_k+1, self.encoder_config.hidden_size)
        for k in range(len(logits)):
            pred_topk_ids = logits[k].argsort()[-self.top_k:].detach().cpu().tolist()
            pred_topk_ids.append(answer[k])
            preds_top.append(pred_topk_ids)
            for l in range(self.top_k):
                ans_embeds[k, l] = self.ans_embeds_all[pred_topk_ids[l]]
                if pred_topk_ids[l] == answer[k]:
                    target.append(1)
                else:
                    target.append(0)
            ans_embeds[k, -1] = self.ans_embeds_all[answer[k]]
            target.append(1)
        
        ans_embeds = ans_embeds.to(self.device)
        target = torch.tensor(target, dtype=torch.long).view(ans_embeds.shape[0], -1)
        return  ans_embeds, target, preds_top
    
    def get_negative_pair(self, answer_embeds, similarity_matrix, target):
        dist = similarity_matrix
        dist[target.bool()] = -1

        dist_max = dist.max(1)  # 取最大值
        negative = torch.zeros(len(dist), 768).to(self.device)
        for i in range(len(dist)):
            negative[i] = answer_embeds[i, dist_max.indices[i], :]
        return negative 


class Bistep_vqa_pretrain(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:1",
                 top_k = 10,
                 temperature = 1,
                 mlm_probability = 0.15,
                 config = None
                 ):
        super().__init__()
        self.device = device
        self.top_k = top_k
        self.output_size = output_size
        self.temperature = temperature
        self.mlm_probability = mlm_probability
        self.encoder_config = BertConfig.from_json_file(med_config)
        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.sentence_tokens = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.sentence_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.sentence_bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # self.gru = nn.GRU(input_size=self.encoder_config.hidden_size, hidden_size=self.encoder_config.hidden_size, num_layers=2)
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt')
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel(config=modelConfig)
        
        # self.transfor1 = Encoder(self.encoder_config.hidden_size, self.encoder_config.n_layers, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        # self.transfor2 = Encoder(self.encoder_config.hidden_size, self.encoder_config.n_layers, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        self.transfor3 = Text_base_attention2(self.encoder_config.hidden_size, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)

        self.classifier1 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        # self.classifier2 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        
        # self.classifier3 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        
        # self.classifier_open_close = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_triplet= nn.TripletMarginLoss(margin=1.0, p=2)
        self.multi_label_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
        
        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.queue_size = self.encoder_config.queue_size
        self.momentum = self.encoder_config.momentum
        self.itm_head = nn.Linear(self.encoder_config.hidden_size, 2) 
        
        # create momentum models
        self.visual_encoder_m, _ = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.text_encoder_m = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 
        # self.transfor3_m = Text_base_attention2(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        # self.classifier1_m = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        self.model_pairs = [[self.vit_encoder, self.visual_encoder_m],
                            [self.text_encoder, self.text_encoder_m],
                            # [self.transfor3, self.transfor3_m],
                            # [self.classifier1, self.classifier1_m],
                           ]
        self.copy_params()
        
        # create the queue
        self.register_buffer("image_queue", torch.randn(self.encoder_config.hidden_size, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.encoder_config.hidden_size, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) 
        
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
    
    def forward(self, image, tokens, segment_ids, input_mask, text=None, alpha=0.2,):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
            
        v_feat = self.vit_encoder(image)
        w_feat = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        s_feat = torch.tensor(self.sentence_bert.encode(text, device=self.device)).to(self.device)      

        ################### 
        image_feat = F.normalize(v_feat[:,0,:], dim=-1)
        text_feat = F.normalize(w_feat[:,0,:], dim=-1)
       
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(image_embeds_m[:,0,:],dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)
            text_output_m = self.text_encoder_m(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
            text_feat_m = F.normalize(text_output_m[:,0,:],dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
            
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)
            
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            
        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp
        
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
        loss_ita = (loss_i2t+loss_t2i)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)    
        
        outputs1, x_v1, x_v2 = self.transfor3(v_feat, w_feat, s_feat.unsqueeze(1))
        # outputs1 = outputs1[:, 0, :]
        # outputs1 = self.ff1(outputs1)
        
        ##################3 ITM
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:, :bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs],dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)
            
        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(v_feat[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        sentence_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(w_feat[neg_idx])
            sentence_neg.append(s_feat[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)   
        sentence_neg = torch.stack(sentence_neg, dim=0) 
        
        text_embeds_all = torch.cat([w_feat, text_embeds_neg], dim=0)    
        sentence_embeds_all = torch.cat([s_feat, sentence_neg], dim=0)     
        image_embeds_all = torch.cat([image_embeds_neg, v_feat], dim=0)
        output_neg, _, _ = self.transfor3(image_embeds_all, text_embeds_all, sentence_embeds_all.unsqueeze(1))
        
        vl_embeddings = torch.cat([outputs1[:,0,:], output_neg[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                            dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels) 
        
        
        ##================= MLM ========================##
        input_ids = tokens.clone()
        labels = input_ids.clone()  # token_ids
        
        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.output_size, self.device, targets=labels,
                                      probability_matrix = probability_matrix)
        
        sentence_input = self.sentence_tokens(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        sentence_input_ids = sentence_input.input_ids.clone()
        senrence_labels = sentence_input_ids.clone()
        sentence_matrix = torch.full(senrence_labels.shape, self.mlm_probability) 
        sentence_input_ids, _ = self.mask(sentence_input_ids, self.output_size, self.device, targets=senrence_labels,
                                      probability_matrix = sentence_matrix)
        sentence_input["input_ids"] = sentence_input_ids
        
        # with torch.no_grad():
        #     w_feat_mask_m = self.text_encoder_m(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        #     s_output_mask_m = self.sentence_model(**sentence_input)
        #     sentence_embeddings_m = self.mean_pooling(s_output_mask_m, sentence_input['attention_mask'])
        #     # Normalize embeddings
        #     sentence_embeddings_m = F.normalize(sentence_embeddings_m, p=2, dim=1)
        #     outputs_m, _, _ = self.transfor3_m(v_feat, w_feat_mask_m, sentence_embeddings_m.unsqueeze(1))
        #     logits_m = self.classifier1_m(outputs_m)
        
        w_feat_mask = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        s_output_mask = self.sentence_model(**sentence_input)
        sentence_embeddings = self.mean_pooling(s_output_mask, sentence_input['attention_mask'])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        outputs, _, _ = self.transfor3(v_feat, w_feat_mask, sentence_embeddings.unsqueeze(1))
        logits = self.classifier1(outputs)
        
        masked_lm_loss = self.loss_ce(logits.view(-1, self.output_size), labels.view(-1))
        # loss_distill = -torch.sum(F.log_softmax(logits, dim=-1)*F.softmax(logits_m,dim=-1), dim=-1)
        # loss_distill = loss_distill[labels!=-100].mean()
        # masked_lm_loss = (1-alpha)*masked_lm_loss + alpha*loss_distill
        
        loss = masked_lm_loss + loss_itm + loss_ita

        return logits, loss, labels
    
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient
    
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)
        image_feats = image_feat
        text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def seperate(v, q, a, input_mask, answer_target, qid):     #q: b x 12 x 1024  v:  b x 1 x 128 answer_target : 1 x b
    indexs_open = []
    indexs_close = []
    for i in range(len(answer_target)):
        if answer_target[i]==0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)

    return v[indexs_open,:,:], v[indexs_close,:,:], q[indexs_open,:,:],\
           q[indexs_close,:,:], input_mask[indexs_open,:], input_mask[indexs_close,:],\
               a[indexs_open], a[indexs_close], qid[indexs_open], qid[indexs_close]


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]) 

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(Transformer_Encoder(d_model, heads, dropout), N)
        
    def forward(self, v_feat, w_feat, s_feat):
        for i in range(self.N):
            v_feat, w_feat, s_feat = self.layers[i](v_feat, w_feat, s_feat)
        return v_feat, w_feat, s_feat


class CMF(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__() 
        self.gobal_attention = MultiHeadAttention(heads, d_model, dropout=dropout)
    
    def forward(self, q, k):
        socre = torch.matmul(q, k.transpose(-2, -1))    # bs * ql * vl
        kg_output1, k1 = torch.topk(socre, dim=-1, k=1) # bs * 362 * 1
        hard_attention_value1 = gumbel_softmax(kg_output1.squeeze()) # bs * 362
        head1 = (q * hard_attention_value1.unsqueeze(-1)).sum(-2) # bs * 768
        head = self.gobal_attention(k, head1.unsqueeze(1), head1.unsqueeze(1))
        return head

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) # [4, 12, 20, 1]
    if mask is not None:
        # mask = mask[:, None, None, :].float()
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)    # [4, 12, 20, 64]
    return output, scores 

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)
        
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))
    
class PositionWiseFeedForward2(nn.Module):
    def __init__(self, hidden_size):
        super(PositionWiseFeedForward2,self).__init__()
        self.fc1 = nn.Linear(hidden_size*2, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)
        
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class PCSI(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.w_linear1 = nn.Linear(d_model, d_model)
        self.w_linear2 = nn.Linear(d_model, d_model)
        self.v_linear1 = nn.Linear(d_model, d_model)
        self.v_linear2 = nn.Linear(d_model, d_model)
        self.v_linear3 = nn.Linear(d_model, d_model)
        self.v_linear4 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, w_feat, v_feat):
        bs = w_feat.size(0)
        w1 = self.w_linear1(w_feat).view(bs, -1, self.h, self.d_k)
        w2 = self.w_linear2(w_feat).view(bs, -1, self.h, self.d_k)
        v1 = self.v_linear1(v_feat).view(bs, -1, self.h, self.d_k)
        v2 = self.v_linear2(v_feat).view(bs, -1, self.h, self.d_k)
        v3 = self.v_linear3(v_feat).view(bs, -1, self.h, self.d_k)
        v4 = self.v_linear4(v_feat).view(bs, -1, self.h, self.d_k)
        
        w1 = w1.transpose(1,2)
        w2 = w2.transpose(1,2)
        v1 = v1.transpose(1,2)
        v2 = v2.transpose(1,2)
        v3 = v3.transpose(1,2)
        v4 = v4.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = torch.matmul(v1, v2.transpose(-2, -1)) + F.softmax(torch.matmul(w1, v3.transpose(-2, -1)),  dim=-1).sum(-2).unsqueeze(2)
        scores = F.softmax(scores, dim=-1)
        
        scores = self.drop(scores)
        
        scores = torch.matmul(scores, v4)   # [4, 12, 20, 64]
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class Sentence_word_Attention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.w_linear1 = nn.Linear(d_model, d_model)
        self.w_linear2 = nn.Linear(d_model, d_model)
        self.w_linear3 = nn.Linear(d_model, d_model)
        self.w_linear4 = nn.Linear(d_model, d_model)
        self.s_linear3 = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, word, sent, mask=None):
        
        bs = word.size(0)
        # perform linear operation and split into N heads
        w1 = self.w_linear1(word).view(bs, -1, self.h, self.d_k)
        w2 = self.w_linear2(word).view(bs, -1, self.h, self.d_k)
        w3 = self.w_linear3(word).view(bs, -1, self.h, self.d_k)
        w4 = self.w_linear4(word).view(bs, -1, self.h, self.d_k)
        s = self.w_linear3(sent).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        w1 = w1.transpose(1,2)
        w2 = w2.transpose(1,2)
        w3 = w3.transpose(1,2)
        w4 = w4.transpose(1,2)
        s = s.transpose(1,2)
        
        # calculate attention using function we will define next
        # # [4, 12, 20, 20] + [4, 12, 20, 1]
        scores = torch.matmul(w1, w2.transpose(-2, -1)) + torch.matmul(w3, s.transpose(-2, -1))
        scores = F.softmax(scores, dim=-1)
        
        scores = self.dropout(scores)

        scores = torch.matmul(scores, w4)   # [4, 12, 20, 64]
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output
        

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None, return_score=False):
        
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)     # [4, 1, 12, 64]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)     # [4, 20, 12, 64]
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)     # [4, 1, 12, 64]
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)    # [4, 12, 1, 64]
        q = q.transpose(1,2)    # [4, 12, 20, 64]
        v = v.transpose(1,2)    # [4, 12, 1, 64]
        
        # calculate attention using function we will define next
        scores, attn_score = attention(q, k, v, self.d_k, mask, self.dropout)   # [4, 12, 20, 64]
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)  # [4, 20, 768]
        output = self.out(concat)
        if return_score:
            return output, attn_score
        return output

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, hidden_size, heads, dropout = 0.1):
        super(MultiHeadedSelfAttention, self).__init__()
        self.proj_q = nn.Linear(hidden_size, hidden_size)
        self.proj_k = nn.Linear(hidden_size, hidden_size)
        self.proj_v = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.scores = None
        self.n_heads = heads
        
    def forward(self, q, k, v, mask):
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h, scores
    
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1  
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class Transformer_Encoder(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.dual_attention = Dual_attention(d_model, heads, dropout=0.1)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = PositionWiseFeedForward(d_model)

    def forward(self, v_feat, w_feat, s_feat):
        x2 = self.norm_1(v_feat)
        x = v_feat + self.dropout_1(self.attn_1(x2, x2, x2))
        x2 = self.norm_2(x)
        
        e_outputs, w_outputs, s_outputs = self.dual_attention(x2, w_feat, s_feat)
        
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs))
        x2 = self.norm_3(x)
        outputs = x + self.dropout_3(self.ff(x2))
        return outputs, w_outputs, s_outputs

class Dual_attention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.norm_4 = nn.LayerNorm(d_model)
        self.norm_5 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        self.ff1 = PositionWiseFeedForward(d_model)
        self.ff2 = PositionWiseFeedForward(d_model)
    
    def forward(self, v_feat, w_feat, s_feat):
        # v_feat: vision feature
        # w_feat: words feature
        # s_feat: sentence feature 
        
        w_feat = self.norm_1(w_feat)
        x = w_feat + self.dropout_1(self.attn_1(w_feat, w_feat, w_feat))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, v_feat, v_feat))
        
        x2 = self.norm_3(x)
        x1 = x + self.dropout_3(self.ff1(x2))   # w_feat output
        
        x = x + self.dropout_4(self.attn_3(x2, s_feat, s_feat)) # v_feat output 
        
        x2 = self.norm_4(x)
        x3 = x + self.dropout_5(self.ff2(x2))   # s_feat output
        
        return x, x1, x3

class Text_base_attention2(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.norm_4 = nn.LayerNorm(d_model)
        self.norm_5 = nn.LayerNorm(d_model)
        self.norm_6 = nn.LayerNorm(d_model)
        self.norm_7 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        self.dropout_6 = nn.Dropout(dropout)
        self.dropout_7 = nn.Dropout(dropout)
        self.dropout_8 = nn.Dropout(dropout)
        
        self.sw_attn = Sentence_word_Attention(heads, d_model, dropout=dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_4 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_5 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_6 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_7 = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        self.cmf1 = CMF(heads, d_model)
        self.cmf2 = CMF(heads, d_model)
        
        self.ff1 = PositionWiseFeedForward(d_model)
        self.ff2 = PositionWiseFeedForward(d_model)
        self.ff3 = PositionWiseFeedForward(d_model)
        self.ff4 = PositionWiseFeedForward(d_model)
        self.ff5 = PositionWiseFeedForward(d_model)
    
    def forward(self, v_feat, w_feat, s_feat, return_score=False):
        # v_feat: vision feature
        # w_feat: words feature
        # s_feat: sentence feature 
        w_feat = self.attn_1(w_feat, s_feat, s_feat)
        x_ws = self.ff5(w_feat)
        # w_feat = self.sw_attn(w_feat, s_feat)
        
        x_v1 = torch.cat((v_feat, s_feat), dim=1)
        x_v = self.norm_1(x_v1)
        x_v1 = x_v1 + self.dropout_1(self.attn_2(x_v, x_v, x_v))
        x_v1 = self.ff1(x_v1)
        
        x_v2 = torch.cat((v_feat, w_feat), dim=1)
        x_v = self.norm_2(x_v2)
        x_v2 = x_v2 + self.dropout_2(self.attn_3(x_v, x_v, x_v))
        x_v2 = self.ff2(x_v2)
        
        x_w1 = self.norm_3(w_feat)
        w_feat = w_feat + self.dropout_3(self.attn_4(x_w1, x_v1, x_v1))
        x_w1 = self.norm_4(w_feat)
        w_feat = w_feat + self.dropout_4(self.attn_5(x_w1, x_w1, x_w1))
        w_feat = self.ff3(w_feat)
        
        x_w2 = self.norm_5(w_feat)
        # w_feat = w_feat + self.dropout_5(self.attn_6(x_w2, x_v2, x_v2, return_score=True))
        x_w, score = self.attn_6(x_w2, x_v2, x_v2, return_score=True)
        w_feat = w_feat + self.dropout_5(x_w)
        x_w2 = self.norm_6(w_feat)
        w_feat = w_feat + self.dropout_6(self.attn_7(x_w2, x_w2, x_w2))
        
        x_wv2 = self.norm_7(w_feat)
        outputs = self.dropout_7(self.ff4(x_wv2))   # s_feat output
        return outputs, x_ws, x_v1, x_v2, score

class Text_base_attention4(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.norm_4 = nn.LayerNorm(d_model)
        self.norm_5 = nn.LayerNorm(d_model)
        self.norm_6 = nn.LayerNorm(d_model)
        self.norm_7 = nn.LayerNorm(d_model)
        self.norm_8 = nn.LayerNorm(d_model)
        self.norm_9 = nn.LayerNorm(d_model)
        self.norm_10 = nn.LayerNorm(d_model)
        self.norm_11 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        self.dropout_6 = nn.Dropout(dropout)
        self.dropout_7 = nn.Dropout(dropout)
        self.dropout_8 = nn.Dropout(dropout)
        self.dropout_9 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_4 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_5 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_6 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_7 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_8 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_9 = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        self.cmf1 = CMF(heads, d_model)
        self.cmf2 = CMF(heads, d_model)
        self.pcsi1 = PCSI(heads, d_model, dropout=dropout)
        self.pcsi2 = PCSI(heads, d_model, dropout=dropout)
        
        self.ff1 = PositionWiseFeedForward(d_model)
        self.ff2 = PositionWiseFeedForward(d_model)
        self.ff3 = PositionWiseFeedForward(d_model)
        self.ff4 = PositionWiseFeedForward(d_model)
        self.ff5 = PositionWiseFeedForward(d_model)
        self.ff6 = PositionWiseFeedForward2(d_model)
    
    def forward(self, v_feat, w_feat, s_feat):
        # v_feat: vision feature
        # w_feat: words feature
        # s_feat: sentence feature 
        w_feat = self.attn_1(w_feat, s_feat, s_feat)
        x_ws = self.ff5(w_feat)
        
        x_v1 = torch.cat((v_feat, s_feat), dim=1)
        x_v = self.norm_1(x_v1)
        x_v1 = x_v1 + self.dropout_1(self.attn_2(x_v, x_v, x_v))
        x_v1 = self.ff1(x_v1)
        
        x_v2 = torch.cat((v_feat, w_feat), dim=1)
        x_v = self.norm_2(x_v2)
        x_v2 = x_v2 + self.dropout_2(self.attn_3(x_v, x_v, x_v))
        x_v2 = self.ff2(x_v2)
        
        # x_w1 = self.norm_3(w_feat)
        # w_feat = w_feat + self.dropout_3(self.attn_4(x_w1, x_v1, x_v1))
        # x_w1 = self.norm_4(w_feat)
        # w_feat = w_feat + self.dropout_4(self.attn_5(x_w1, x_w1, x_w1))
        # w_feat = self.ff3(w_feat)
        
        # x_w2 = self.norm_5(w_feat)
        # w_feat = w_feat + self.dropout_5(self.attn_6(x_w2, x_v2, x_v2))
        # x_w2 = self.norm_6(w_feat)
        # w_feat = w_feat + self.dropout_6(self.attn_7(x_w2, x_w2, x_w2))
        
        # x_w3 = self.norm_7(w_feat)
        # outputs = self.dropout_7(self.ff4(x_w3))   # s_feat output
        
        ##################################
        ########## test1 70
        # x_w1 = self.norm_3(w_feat)
        # w_feat1 = w_feat + self.dropout_3(self.cmf1(x_v1, x_w1))
        # w_feat2 = w_feat + self.dropout_3(self.cmf2(x_v2, x_w1))
        # x_w1 = self.norm_4(w_feat1)
        # x_w2 = self.norm_5(w_feat2)
        # w_feat = w_feat + self.dropout_5(self.attn_6(x_w1, x_w2, x_w2))
        # # w_feat3 = w_feat1 + w_feat2
        # x_w3 = self.norm_7(w_feat)
        # outputs = self.dropout_7(self.ff4(x_w3))   # s_feat output
        
        ########### test2 68
        # x_vsw = torch.cat((x_v1, w_feat), dim=1)
        # x_w1 = self.norm_3(w_feat)
        # w_feat = w_feat + self.dropout_3(self.attn_4(x_w1, x_vsw, x_vsw))
        # x_w1 = self.norm_4(w_feat)
        # w_feat = w_feat + self.dropout_4(self.attn_5(x_w1, x_w1, x_w1))
        # w_feat = self.ff3(w_feat)
        # x_w3 = self.norm_7(w_feat)
        # outputs = self.dropout_7(self.ff4(x_w3))   # s_feat output
        
        ########### test3
        # x_v1 = self.norm_3(x_v1)
        # x_v2 = self.norm_4(x_v2)
        # w_feat = self.norm_5(w_feat)
        # w_feat =self.pcsi1(x_v1, w_feat)
        # w_feat = self.norm_6(w_feat)
        # w_feat =self.pcsi2(x_v2, w_feat)
        # x_w3 = self.norm_7(w_feat)
        # outputs = self.dropout_7(self.ff4(x_w3))
        
        #################test 4 
        # Cross-attention centered on wv
        # x_wv = self.norm_3(x_v2)
        # # x_wv2 = x_v2 + self.dropout_3(self.attn_4(x_wv, w_feat, w_feat))
        # x_wv2 = x_v2 + self.dropout_3(self.pcsi1(w_feat, x_wv))
        # x_wv = self.norm_4(x_wv2)
        # x_wv2 = x_wv2 + self.dropout_4(self.attn_5(x_wv, x_wv, x_wv))
        # x_wv2 = self.ff3(x_wv2)
        
        # # x_wv2 = x_sv1
        # x_wv = self.norm_5(x_wv2)
        # # x_wv2 = x_wv2 + self.dropout_5(self.attn_6(x_wv, x_v1, x_v1))
        # x_wv2 = x_wv2 + self.dropout_5(self.pcsi2(x_v1, x_wv))
        # x_wv = self.norm_6(x_wv2)
        # x_wv2 = x_wv2 + self.dropout_6(self.attn_7(x_wv, x_wv, x_wv))
        
        # x_wv2 = self.norm_7(x_wv2)
        # outputs = self.dropout_7(self.ff4(x_wv2))   # s_feat output
        
        
        x_w1 = self.norm_3(w_feat)
        x_w3 = w_feat + self.dropout_3(self.attn_4(x_w1, x_v1, x_v1))
        x_w1 = self.norm_4(x_w3)
        x_w3 = x_w3 + self.dropout_4(self.attn_5(x_w1, x_w1, x_w1))
        # x_w3 = self.ff3(x_w3)
        
        x_w2 = self.norm_5(w_feat)
        w_feat = w_feat + self.dropout_5(self.attn_6(x_w2, x_v2, x_v2))
        x_w2 = self.norm_6(w_feat)
        w_feat = w_feat + self.dropout_6(self.attn_7(x_w2, x_w2, x_w2))
        
        x_w2 = self.norm_8(w_feat)
        x_w3 = self.norm_10(x_w3)
        w_feat = w_feat + self.dropout_7(self.attn_8(x_w2, x_w3, x_w3))
        x_w2 = self.norm_9(w_feat)
        w_feat = w_feat + self.dropout_8(self.attn_9(x_w2, x_w2, x_w2))
        
        x_wv2 = self.norm_11(w_feat)
        outputs = self.dropout_9(self.ff4(x_wv2))   # s_feat output
        
        return outputs, x_ws, x_v1, x_v2


class Text_base_attention3(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.norm_4 = nn.LayerNorm(d_model)
        self.norm_5 = nn.LayerNorm(d_model)
        self.norm_6 = nn.LayerNorm(d_model)
        self.norm_7 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        self.dropout_6 = nn.Dropout(dropout)
        self.dropout_7 = nn.Dropout(dropout)
        self.dropout_8 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_4 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_5 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_6 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_7 = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        self.pcsi1 = PCSI(heads, d_model, dropout=dropout)
        self.pcsi2 = PCSI(heads, d_model, dropout=dropout)
        
        self.ff1 = PositionWiseFeedForward(d_model)
        self.ff2 = PositionWiseFeedForward(d_model)
        self.ff3 = PositionWiseFeedForward(d_model)
        self.ff4 = PositionWiseFeedForward(d_model)
        self.ff5 = PositionWiseFeedForward(d_model)
    
    def forward(self, v_feat, w_feat, s_feat):
        # v_feat: vision feature
        # w_feat: words feature
        # s_feat: sentence feature 
        w_feat = self.attn_1(w_feat, s_feat, s_feat)
        
        x_sv1 = torch.cat((s_feat, v_feat), dim=1)
        x_sv = self.norm_1(x_sv1)
        x_sv1 = x_sv1 + self.dropout_1(self.attn_2(x_sv, x_sv, x_sv))
        x_sv1 = self.ff1(x_sv1)
        
        x_wv1 = torch.cat((w_feat, v_feat), dim=1)
        x_wv = self.norm_2(x_wv1)
        x_wv1 = x_wv1 + self.dropout_2(self.attn_3(x_wv, x_wv, x_wv))
        x_wv1 = self.ff2(x_wv1)
        
        # Cross-attention centered on wv
        x_wv = self.norm_3(x_wv1)
        x_wv2 = x_wv1 + self.dropout_3(self.attn_4(x_wv, w_feat, w_feat))
        x_wv = self.norm_4(x_wv2)
        x_wv2 = x_wv2 + self.dropout_4(self.attn_5(x_wv, x_wv, x_wv))
        x_wv2 = self.ff3(x_wv2)
        
        # x_wv2 = x_sv1
        x_wv = self.norm_5(x_wv2)
        x_wv2 = x_wv2 + self.dropout_5(self.attn_6(x_wv, x_sv1, x_sv1))
        x_wv = self.norm_6(x_wv2)
        x_wv2 = x_wv2 + self.dropout_6(self.attn_7(x_wv, x_wv, x_wv))
        
        x_wv2 = self.norm_7(x_wv2)
        outputs = self.dropout_7(self.ff4(x_wv2))   # s_feat output
        
        return outputs, w_feat, x_sv1, x_wv1
    
class Text_base_attention5(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.norm_4 = nn.LayerNorm(d_model)
        self.norm_5 = nn.LayerNorm(d_model)
        self.norm_6 = nn.LayerNorm(d_model)
        self.norm_7 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        self.dropout_5 = nn.Dropout(dropout)
        self.dropout_6 = nn.Dropout(dropout)
        self.dropout_7 = nn.Dropout(dropout)
        self.dropout_8 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_4 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_5 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_6 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_7 = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        self.pcsi1 = PCSI(heads, d_model, dropout=dropout)
        self.pcsi2 = PCSI(heads, d_model, dropout=dropout)
        
        self.ff1 = PositionWiseFeedForward(d_model)
        self.ff2 = PositionWiseFeedForward(d_model)
        self.ff3 = PositionWiseFeedForward(d_model)
        self.ff4 = PositionWiseFeedForward(d_model)
        self.ff5 = PositionWiseFeedForward(d_model)
    
    def forward(self, v_feat, w_feat, s_feat):
        # v_feat: vision feature
        # w_feat: words feature
        # s_feat: sentence feature 
        w_feat = self.attn_1(w_feat, s_feat, s_feat)
        
        # x_sv1 = torch.cat((s_feat, v_feat), dim=1)
        # x_sv = self.norm_1(x_sv1)
        # x_sv1 = x_sv1 + self.dropout_1(self.attn_2(x_sv, x_sv, x_sv))
        # x_sv1 = self.ff1(x_sv1)
        
        # x_wv1 = torch.cat((w_feat, v_feat), dim=1)
        # x_wv = self.norm_2(x_wv1)
        # x_wv1 = x_wv1 + self.dropout_2(self.attn_3(x_wv, x_wv, x_wv))
        # x_wv1 = self.ff2(x_wv1)
        
        # Cross-attention centered on sv
        # x_sv = self.norm_3(x_sv1)
        # x_sv2 = x_sv1 + self.dropout_3(self.attn_4(x_sv, w_feat, w_feat))
        # x_sv = self.norm_4(x_sv2)
        # x_sv2 = x_sv2 + self.dropout_4(self.attn_5(x_sv, x_sv, x_sv))
        # x_sv2 = self.ff3(x_sv2)
        
        # x_wv2 = x_sv1
        # x_sv = self.norm_5(x_sv2)
        # x_sv2 = x_sv2 + self.dropout_5(self.attn_6(x_sv, x_wv1, x_wv1))
        # x_sv = self.norm_6(x_sv2)
        # x_sv2 = x_sv2 + self.dropout_6(self.attn_7(x_sv, x_sv, x_sv))
        
        # x_sv2 = self.norm_7(x_sv2)
        # outputs = self.dropout_7(self.ff4(x_sv2))   # s_feat output
        
        ######### vs
        # x_ws = self.norm_3(w_feat)
        # x_ws2 = x_ws + self.dropout_3(self.attn_4(x_ws, x_sv1, x_sv1))
        # x_ws = self.norm_4(x_ws2)
        # x_ws2 = x_ws2 + self.dropout_4(self.attn_5(x_ws, x_ws, x_ws))
        # x_ws2 = self.ff3(x_ws2)
        
        ############ vw
        # x_wv2 = w_feat
        # x_wv = self.norm_5(x_wv2)
        # x_wv2 = x_wv2 + self.dropout_5(self.attn_6(x_wv, x_wv1, x_wv1))
        # x_wv = self.norm_6(x_wv2)
        # x_wv2 = x_wv2 + self.dropout_6(self.attn_7(x_wv, x_wv, x_wv))
        
        # x_wv2 = self.norm_7(x_wv2)
        # outputs = self.dropout_7(self.ff4(x_wv2))   # s_feat output
        
        x_sv = self.norm_1(v_feat)
        x_sv1 = x_sv1 + self.dropout_1(self.attn_2(x_sv, s_feat, s_feat))
        x_sv1 = self.ff1(x_sv1)
        x_wv1 = x_wv1 + self.dropout_2(self.attn_3(x_sv, w_feat, w_feat))
        x_wv1 = self.ff2(x_wv1)
        
        # x_sv = self.norm_3(x_sv1)
        # x_sv2 = w_feat + self.dropout_3(self.attn_4(w_feat, x_sv, x_sv))
        # x_sv = self.norm_4(x_sv2)
        # x_sv2 = x_sv2 + self.dropout_4(self.attn_5(x_sv, x_sv, x_sv))
        # x_sv2 = self.ff3(x_sv2)
        
        # x_sv = self.norm_5(x_sv2)
        # x_sv2 = x_sv2 + self.dropout_5(self.attn_6(x_sv, x_wv1, x_wv1))
        # x_sv = self.norm_6(x_sv2)
        # x_sv2 = x_sv2 + self.dropout_6(self.attn_7(x_sv, x_sv, x_sv))
        x_v = self.norm_2(x_sv1) + self.norm_3(x_wv1)
        x_ws = self.norm_3(w_feat)
        x_wsv = torch.cat((x_ws, x_v), dim=1)
        x_sv2 = self.dropout_5(self.attn_6(x_wsv, x_wsv, x_wsv))
        
        x_sv2 = self.norm_7(x_sv2)
        outputs = self.dropout_7(self.ff4(x_sv2))   # s_feat output
        
        return outputs, w_feat, x_sv1, x_wv1
    
class Transformer_Encoder1(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.norm_4 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_4 = MultiHeadAttention(heads, d_model, dropout=dropout)
        
        self.ff1 = PositionWiseFeedForward(d_model)
        self.ff2 = PositionWiseFeedForward(d_model)
        self.ff3 = PositionWiseFeedForward(d_model)
        self.ff4 = PositionWiseFeedForward(d_model)
        
    def forward(self, input1, input2, input3):
        z1 = torch.cat((input1, input2), dim=1)
        
        z = self.norm_1(z1)
        z1 = z1 + self.dropout_1(self.attn_3(z, z, z))
        z1 = self.ff2(z1)
        
        z = self.norm_2(z1)
        z2 = z1 + self.dropout_2(self.attn_2(z, input3, input3))
        z = self.norm_3(z2)
        z2 = z2 + self.dropout_3(self.attn_3(z, z, z))
        
        z2 = self.ff2(z2)
        
        return z2
    
class Transfors(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
    
        self.encoder1 = Transformer_Encoder1(d_model, heads, dropout)
        self.encoder2 = Transformer_Encoder1(d_model, heads, dropout)
        self.encoder3 = Transformer_Encoder1(d_model, heads, dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
        
    def forward(self, v_feat, w_feat, s_feat):
        # x_vw = self.encoder1(v_feat, w_feat, s_feat)
        # x_vs = self.encoder2(v_feat, s_feat, w_feat)
        x_ws = self.encoder3(w_feat, s_feat, v_feat)
        
        # x_vw2 = self.attn_1(x_ws, x_vw, x_vw)
        # x_vs2 = self.attn_1(x_ws, x_vs, x_vs)
        # output = self.attn_1(x_vw2, x_vs2, x_vs2)
        output = self.ff1(x_ws)
        
        # return output, x_ws, x_vs, x_vw
        return output, x_ws, x_ws, x_ws
        