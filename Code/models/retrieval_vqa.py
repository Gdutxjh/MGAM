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

from models.med import BertConfig
from models.Attn import MultiHeadAttention
from models.vit import VisionTransformer

class retrieval_vqa(nn.Module):
    def __init__(self, 
                image_size = 448,
                med_config = 'configs/med_config.json',
                vit = 'base',
                vit_grad_ckpt = False,
                vit_ckpt_layer = 0,
                output_size = 1024,
                device = "cuda:1",
                top_k = 10,
                temperature = 1,
                config = None,
                text_cfg=None,
                ):
        super().__init__()
        self.topk = config["topk"]
        self.device = device
        self.context_length = config["context_length"]
        self.question_length = config["question_length"]
        # text_encoder
        self.encoder_config = BertConfig.from_json_file(med_config)
        # modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        # self.bert_encoder = BertModel(config=modelConfig)
        self.sentence_bert = SentenceTransformer('./pre_model/mpnet-base', device=self.device)
        
        # text_cfg = CLIPTextCfg(text_cfg)
        text_cfg["bert_model_name"] = '../model/BiomedNLP-BiomedBERT-base-uncased'
        tokenizer_name = text_cfg["bert_model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.retrieval_encoder = AutoModel.from_pretrained(text_cfg["bert_model_name"])
        self.question_encoder = AutoModel.from_pretrained(text_cfg["bert_model_name"])
        
        self.vit_encoder1, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.vit_encoder2, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size2, vit_grad_ckpt, vit_ckpt_layer)
        
        
        self.encoder1 = Base_Encoder(self.encoder_config.heads, self.encoder_config.hidden_size, 
                                     dropout=self.encoder_config.hidden_dropout_prob)
        self.encoder2 = Base_Encoder(self.encoder_config.heads, self.encoder_config.hidden_size, 
                                     dropout=self.encoder_config.hidden_dropout_prob)
        
        
        self.fusion1 = Text_base_Transformer(self.encoder_config.hidden_size, self.encoder_config.heads, 
                    self.encoder_config.hidden_dropout_prob)
        
        
        self.ff1 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, output_size))
        self.ff2 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, output_size))
        self.ff3 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.norm = nn.LayerNorm(self.encoder_config.hidden_size)
        self.loss_ce = nn.CrossEntropyLoss()
        self.KL_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        
        
    def encode_retrieval_text(self, retrieval_text):
        encoded_input = self.tokenizer(retrieval_text, padding='max_length', truncation=True, max_length=self.context_length, return_tensors='pt')
        encoded_input['input_ids'] = encoded_input['input_ids'].to(self.device)  # [128, 77]

        x = self.retrieval_encoder(
            input_ids = encoded_input['input_ids'],
            output_attentions = False
        )
        x = x['last_hidden_state']
        # last_token_index = torch.nonzero( (encoded_input['input_ids'] == self.cls_id).squeeze() )
        # text_features = x[torch.arange(x.shape[0]), last_token_index[:, 1]]
        text_features = x[:, 0]
        # text_features = text_features @ self.text_projection  # NOTE for matching
        
        return text_features
        
    def forward(self, image, question, retrieval_text, answer_id):
        bs = image.shape[0]     # 4
        nh = self.encoder_config.heads
        w_featmap=image.shape[-2] //self.encoder_config.patch_size    # 3
        h_featmap=image.shape[-1] //self.encoder_config.patch_size    # 3
        
        
        ques_input = self.tokenizer(question, padding='max_length', truncation=True, max_length=self.question_length, return_tensors='pt')
        ques_input['input_ids'] = ques_input['input_ids'].to(self.device)
        ques_feature = self.retrieval_encoder(input_ids = ques_input['input_ids'], output_attentions = False)
        w_feat = ques_feature['last_hidden_state']
        
        s_feat = torch.tensor(self.sentence_bert.encode(question, device=self.device)).to(self.device)
        text_features = self.encode_retrieval_text(retrieval_text)
        text_features = text_features.reshape(bs, self.topk, -1)
        
        v_feat1 = self.vit_encoder1(image)  # [4, 10, 768]
        v_feat2 = self.vit_encoder2(image)  # [4, 197, 768]
        
        ################
        z_v1s, coarse_mask = self.encoder1(v_feat1, s_feat.unsqueeze(1), return_mask=True)   # [2, 12, 11, 11]
        z_v2w, fine_mask = self.encoder2(v_feat2, w_feat, return_mask=True) # [2, 12, 57, 57]
        
        coarse_score = coarse_mask[:, :, 0, 1:-1].reshape(bs, nh, w_featmap, h_featmap)
        upsample_size = self.encoder_config.patch_size//self.encoder_config.patch_size2
        attentions_mask = nn.functional.interpolate(coarse_score.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        attentions_mask = attentions_mask.reshape(bs, nh, -1)
        attentions_mask = torch.cat((coarse_mask[:, :, 0, 0].unsqueeze(-1), attentions_mask, torch.ones(bs, nh, len(ques_input['input_ids'][0])).to(self.device)), dim=-1)

        # coarse上采样
        a = z_v1s[:, 1:-1, :].transpose(1, 2) # [bs, 51, 768]->[bs, 49, 768]->[bs, 768, 49]
        b = a.reshape(bs, self.encoder_config.hidden_size, w_featmap, h_featmap) # [bs, 768, 49]->[bs, 768, 7, 7]
        # [bs, 768, 7, 7] -> [bs, 768, 1, 7, 7]->[bs, 768, 2, 14, 14]->[bs, 768, 14, 14]
        c = nn.functional.interpolate(b.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        d = c.reshape(bs, self.encoder_config.hidden_size, -1).transpose(1, 2)  # [bs, 768, 14, 14]->[bs, 768, 196]->[bs, 196, 768]
        v_coarse = torch.cat((z_v1s[:, 0, :].unsqueeze(1), d, z_v1s[:, -1, :].unsqueeze(1)), dim=1)   # [bs, 196, 768]->[bs, 197, 768]
        
        # attn_mask1 = fine_mask[:, :, 0, :v_coarse.shape[1]] - attentions_mask[:, :, :v_coarse.shape[1]]
        # attn_mask1 = data_normal_2d(attn_mask1, dim=-1, device=self.device)
        # attn_mask1 = torch.cat((attn_mask1, torch.ones(bs, nh, len(tokens[0])-1).to(self.device)), dim=-1)
        # output, z_ws = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask[:, :, :-19], attn_mask1)
        # output, z_ws = self.fusion1(z_v1s, z_v2w, w_feat, s_feat.unsqueeze(1))
        
        output, z_ws, score1, score2 = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), text_features,
                                               attentions_mask, fine_mask[:, :, 0, :])
        
        logits1 = self.ff1(output.mean(1))
        logits2 = self.ff2(z_v1s.mean(1))
        logits3 = self.ff3(z_v2w.mean(1))
        loss_ce1 = self.loss_ce(logits1, answer_id)
        
        target = F.softmax(logits1, dim=1)
        logits2 = F.log_softmax(logits2, dim=1)
        loss_KL2 = self.KL_loss(logits2, target)
        
        logits3 = F.log_softmax(logits3, dim=1)
        loss_KL3 = self.KL_loss(logits3, target)
        loss = loss_ce1 + loss_KL2 + loss_KL3
        
        return logits1, loss
        
        
class Base_Encoder(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
        self.ff2 = PositionWiseFeedForward(d_model)
        
    def forward(self, v_feat, t_feat, return_mask=False):
        z_v1s = torch.cat((v_feat, t_feat), dim=1)
        z_vs = self.norm_1(z_v1s)
        # z_v1s = z_v1s + self.dropout_1(self.attn_2(z_vs, z_vs, z_vs))
        z_vs, attn_mask = self.attn_2(z_vs, z_vs, z_vs, return_mask=return_mask)
        z_v1s = z_v1s + self.dropout_1(z_vs)
        z_v1s = self.ff2(z_v1s)
        if return_mask:
            return z_v1s, attn_mask
        return z_v1s  


class Cross_Encoder(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
        self.ff2 = PositionWiseFeedForward(d_model)
        
    def forward(self, t_feat, v_feat, attn_mask=None):
        z_t = self.norm_1(t_feat)
        z_t, score = self.attn_2(z_t, v_feat, v_feat, mask=attn_mask, return_mask=True)
        t_feat = t_feat + self.dropout_1(z_t)
        # t_feat = t_feat + self.dropout_1(self.attn_2(z_t, v_feat, v_feat, mask=attn_mask))
        z_t = self.norm_2(t_feat)
        t_feat = t_feat + self.dropout_2(self.attn_2(z_t, z_t, z_t))
        t_feat = self.ff2(t_feat)
        
        return t_feat, score

class Text_base_Transformer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.norm_4 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
        
        self.encoder1 = Base_Encoder(heads, d_model, dropout)
        
        self.cross_encoder1 = Cross_Encoder(heads, d_model, dropout=dropout)
        self.cross_encoder3 = Cross_Encoder(heads, d_model, dropout=dropout)
        
        self.cross_encoder2 = Cross_Encoder(heads, d_model, dropout=dropout)
        self.cross_encoder4 = Cross_Encoder(heads, d_model, dropout=dropout)
    
    def forward(self, z_v1s, z_v2w, w_feat, s_feat, retrieval_text, coarse_mask=None, fine_mask=None):
        # z_ws = w_feat + self.attn_1(w_feat, s_feat, s_feat)
        # z_w = self.norm_3(w_feat)
        # z_s = self.norm_4(s_feat)
        # z_ws = w_feat + self.attn_1(z_w, z_s, z_s)
        
        z_ws = self.attn_1(w_feat, s_feat, s_feat)
        z_ws = self.ff1(z_ws+self.norm_4(w_feat))
        # z_ws, mask_ws = self.encoder1(w_feat, s_feat, return_mask=True)

        z_v1s = self.norm_1(z_v1s)
        z_v2w = self.norm_2(z_v2w)
        # z_v1s[:, :-1, :] += z_v2w[:, :-20, :]
        # z_f1 = self.cross_encoder1(z_ws, z_v1s, attn_mask=coarse_mask)
        # z_f2 = self.cross_encoder2(z_f1, z_v2w, attn_mask=fine_mask)
        
        ####baseline
        z_v2w[:, :-19, :] += z_v1s
        # # z_f1 = self.cross_encoder1(w_feat, z_v2w, attn_mask=coarse_mask)#######3
        z_f1, score1 = self.cross_encoder1(z_ws, z_v2w, attn_mask=coarse_mask)
        z_f1, score1 = self.cross_encoder3(z_f1, retrieval_text)
        
        
        z_f2, score2 = self.cross_encoder2(z_f1, z_v2w, attn_mask=fine_mask)
        z_f2, score2 = self.cross_encoder4(z_f2, retrieval_text)
        
        # z_f2 = z_v2w
        
        ########3
        # z_f1 = self.cross_encoder1(z_ws, z_v1s, attn_mask=fine_mask)
        # z_f2 = self.cross_encoder2(z_f1, z_v2w, attn_mask=coarse_mask)
        #####3
        # z_f1 = self.cross_encoder1(z_ws, z_v2w, attn_mask=coarse_mask)
        # z_f2 = self.cross_encoder2(z_f1, z_v1s, attn_mask=fine_mask)
        # output = self.norm_1(z_f2)
        # output = self.dropout_1(self.ff1(output))
        # return z_f2, w_feat###########
        return z_f2, z_ws, score1, score2

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)
        
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))       

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def create_vit(vit, image_size, patch_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=patch_size, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=patch_size, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width