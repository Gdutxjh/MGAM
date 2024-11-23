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
import scipy.stats

from models.vit import VisionTransformer, interpolate_pos_embed
from models.resnet import resnet152 
from models.med import BertConfig, BertLMHeadModel
from models.Attn import MultiHeadAttention
from models.CoAttn import MFB
from models.ban import BAN

class Bistep_vqa7(nn.Module):
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
                 config = None
                 ):
        super().__init__()
        self.device = device
        self.top_k = top_k
        
        self.temperature = temperature
        self.encoder_config = BertConfig.from_json_file(med_config)
        self.vit_encoder1, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.vit_encoder2, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size2, vit_grad_ckpt, vit_ckpt_layer)
        # self.sentence_bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)
        self.sentence_bert = SentenceTransformer('./pre_model/mpnet-base', device=self.device)
        # self.gru = nn.GRU(input_size=self.encoder_config.hidden_size, hidden_size=self.encoder_config.hidden_size, num_layers=2)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel(config=modelConfig)
        self.vs_encoder = Globe_Transfor(self.encoder_config.hidden_size, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        self.encoder = Local_Transfor(self.encoder_config.hidden_size, \
            self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        
        self.ff1 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, output_size))
        self.ff2 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, output_size))
        self.classifier_open_close = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.loss_ce = nn.CrossEntropyLoss()
        self.multi_label_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
        
    def forward(self, image, tokens, segment_ids, input_mask, answer=None, 
                id2ans=None, ans_tokens_list=None, question=None, ques_target=None, val=None):
        # w, h = image.shape[-2], image.shape[-1]
        bs = image.shape[0]     # 4
        w_featmap=image.shape[-2] //self.encoder_config.patch_size    # 14
        h_featmap=image.shape[-1] //self.encoder_config.patch_size    # 14
        
        v_feat1 = self.vit_encoder1(image)  # [4, 197, 768]
        v_feat2 = self.vit_encoder2(image)  # [4, 197, 768]
        w_feat = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        s_feat = torch.tensor(self.sentence_bert.encode(question, device=self.device)).to(self.device)
        
        z_vs, score_vs = self.vs_encoder(s_feat.unsqueeze(1), v_feat1)
        
        # 注意力矩阵上采样
        score_vs = score_vs[:, :, :2, 1:-1].mean(2).reshape(bs, self.encoder_config.heads, w_featmap, h_featmap)    # [4, 12, 7, 7]
        upsample_size = self.encoder_config.patch_size//self.encoder_config.patch_size2
        attentions_mask = nn.functional.interpolate(score_vs.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        attentions_mask = attentions_mask.reshape(bs, self.encoder_config.heads, -1)
        attentions_mask = torch.cat((score_vs[:, :, 0, 0].unsqueeze(-1), attentions_mask), dim=-1)
        ################3
        a = z_vs[:, 1:-1, :].transpose(1, 2) # [bs, 51, 768]->[bs, 49, 768]->[bs, 768, 49]
        b = a.reshape(bs, self.encoder_config.hidden_size, w_featmap, h_featmap) # [bs, 768, 49]->[bs, 768, 7, 7]
        # [bs, 768, 7, 7] -> [bs, 768, 1, 7, 7]->[bs, 768, 2, 14, 14]->[bs, 768, 14, 14]
        c = nn.functional.interpolate(b.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        d = c.reshape(bs, self.encoder_config.hidden_size, -1).transpose(1, 2)  # [bs, 768, 14, 14]->[bs, 768, 196]->[bs, 196, 768]
        e = torch.cat((z_vs[:, 0, :].unsqueeze(1), d), dim=1)   # [bs, 196, 768]->[bs, 197, 768]
        
        output, z_ws = self.encoder(w_feat, s_feat, e, v_feat2, attentions_mask)
        
        
        logits1 = self.ff1(z_vs.mean(1))
        loss_ce1 = self.loss_ce(logits1, answer)
        
        logits2 = self.ff2(output.mean(1))
        loss_ce2 = self.loss_ce(logits2, answer)
        
        ques_logits = self.classifier_open_close(z_ws.mean(1))
        loss_multi = self.multi_label_loss(ques_logits, ques_target)
        
        logits2 = torch.mul(logits2, ques_logits)
        
        loss =  loss_ce1*0.2 + loss_ce2 + 0.2*loss_multi
        
        if val:
            return logits2
        
        return logits2, loss

class Bistep_vqa8(nn.Module):
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
        self.vit_encoder1, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.vit_encoder2, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size2, vit_grad_ckpt, vit_ckpt_layer)
        # self.sentence_bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)
        self.sentence_bert = SentenceTransformer('./pre_model/mpnet-base', device=self.device)
        # self.gru = nn.GRU(input_size=self.encoder_config.hidden_size, hidden_size=self.encoder_config.hidden_size, num_layers=2)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased')
        self.text_encoder = BertModel(config=modelConfig)
        
        # self.encoder1 = Encoder_Base(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        # self.encoder2 = Encoder_Base(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        # self.encoder3 = Encoder_Ques(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        
        # self.fusion = Fusion(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        
        ##############
        # self.fusion1 = Text_base_attention5(self.encoder_config.hidden_size, self.encoder_config.heads, 
        #             self.encoder_config.hidden_dropout_prob)
        
        ################
        self.encoder1 = Base_Encoder(self.encoder_config.heads, self.encoder_config.hidden_size, 
                                     dropout=self.encoder_config.hidden_dropout_prob)
        self.encoder2 = Base_Encoder(self.encoder_config.heads, self.encoder_config.hidden_size, 
                                     dropout=self.encoder_config.hidden_dropout_prob)
        self.fusion1 = Text_base_attention6(self.encoder_config.hidden_size, self.encoder_config.heads, 
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
        self.classifier_open_close = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        # self.softmax = F.softmax(dim=1)
        # self.log_softmax1 = F.log_softmax(dim=1)
        # self.log_softmax2 = F.log_softmax(dim=1)
        self.norm = nn.LayerNorm(self.encoder_config.hidden_size)
        self.loss_ce = nn.CrossEntropyLoss()
        self.multi_label_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
        self.KL_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        
    def forward(self, image, tokens, segment_ids, input_mask, answer=None, 
                id2ans=None, ans_tokens_list=None, question=None, ques_target=None, val=None):
        
        bs = image.shape[0]     # 4
        nh = self.encoder_config.heads
        w_featmap=image.shape[-2] //self.encoder_config.patch_size    # 3
        h_featmap=image.shape[-1] //self.encoder_config.patch_size    # 3
        
        v_feat1 = self.vit_encoder1(image)  # [4, 10, 768]
        v_feat2 = self.vit_encoder2(image)  # [4, 10, 768]
        w_feat = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        s_feat = torch.tensor(self.sentence_bert.encode(question, device=self.device)).to(self.device)
        
        # z_vs, vs_score = self.encoder1(v_feat1, s_feat, return_score=True)
        # z_vw, vw_score = self.encoder2(v_feat2, w_feat)
        # z_ws = self.encoder3(w_feat, s_feat)
        
        # # 注意力矩阵上采样
        # # attn_mask = self.create_mask(vs_score)
        
        # # score_vs = vs_score[:, :, 1:].reshape(bs, self.encoder_config.heads, w_featmap, h_featmap)    # [4, 12, 7, 7]
        # score_vs = vs_score[:, :, 1:].reshape(bs, self.encoder_config.heads, w_featmap, h_featmap)
        # upsample_size = self.encoder_config.patch_size//self.encoder_config.patch_size2
        # attentions_mask = nn.functional.interpolate(score_vs.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        # attentions_mask = attentions_mask.reshape(bs, self.encoder_config.heads, -1)
        # attentions_mask = torch.cat((vs_score[:, :, 0].unsqueeze(-1), attentions_mask), dim=-1)
        
        # ################3
        # a = z_vs[:, 1:, :].transpose(1, 2) # [bs, 51, 768]->[bs, 49, 768]->[bs, 768, 49]
        # b = a.reshape(bs, self.encoder_config.hidden_size, w_featmap, h_featmap) # [bs, 768, 49]->[bs, 768, 7, 7]
        # # [bs, 768, 7, 7] -> [bs, 768, 1, 7, 7]->[bs, 768, 2, 14, 14]->[bs, 768, 14, 14]
        # c = nn.functional.interpolate(b.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        # d = c.reshape(bs, self.encoder_config.hidden_size, -1).transpose(1, 2)  # [bs, 768, 14, 14]->[bs, 768, 196]->[bs, 196, 768]
        # v_globe = torch.cat((z_vs[:, 0, :].unsqueeze(1), d), dim=1)   # [bs, 196, 768]->[bs, 197, 768]
        
        # output = self.fusion(z_ws, v_globe, z_vw, attentions_mask)
        
        # logits1 = self.ff1(z_vs.mean(1))
        # loss_ce1 = self.loss_ce(logits1, answer)
        
        # logits2 = self.ff2(output.mean(1))
        # loss_ce2 = self.loss_ce(logits2, answer)
        
        # ques_logits = self.classifier_open_close(z_ws.mean(1))
        # loss_multi = self.multi_label_loss(ques_logits, ques_target)
        
        # logits2 = torch.mul(logits2, ques_logits)
        
        # loss =  loss_ce1 + loss_ce2 + loss_multi
        ########################33
        # output, z_ws, z_f1, z_f2 = self.fusion1(v_feat1, v_feat2, w_feat, s_feat.unsqueeze(1))
        # logits1 = self.ff1(output.mean(1))
        # loss_ce1 = self.loss_ce(logits1, answer)
        
        # logits2 = self.ff2(z_f1.mean(1))
        # loss_ce2 = self.loss_ce(logits2, answer)
        
        # logits3 = self.ff3(z_f2.mean(1))
        # loss_ce3 = self.loss_ce(logits3, answer)
        
        # loss = loss_ce1 + loss_ce2 + loss_ce3
        ################
        z_v1s, coarse_mask = self.encoder1(v_feat1, s_feat.unsqueeze(1), return_mask=True)   # [2, 12, 11, 11]
        z_v2w, fine_mask = self.encoder2(v_feat2, w_feat, return_mask=True) # [2, 12, 57, 57]
        
        coarse_score = coarse_mask[:, :, 0, 1:-1].reshape(bs, nh, w_featmap, h_featmap)
        upsample_size = self.encoder_config.patch_size//self.encoder_config.patch_size2
        attentions_mask = nn.functional.interpolate(coarse_score.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        attentions_mask = attentions_mask.reshape(bs, nh, -1)
        attentions_mask = torch.cat((coarse_mask[:, :, 0, 0].unsqueeze(-1), attentions_mask, torch.ones(bs, nh, len(tokens[0])).to(self.device)), dim=-1)

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
        
        output, z_ws, score1, score2 = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), 
                                               attentions_mask, fine_mask[:, :, 0, :-19], return_mask=True)
       
        
        logits1 = self.ff1(output.mean(1))
        logits2 = self.ff2(z_v1s.mean(1))
        logits3 = self.ff3(z_v2w.mean(1))
        loss_ce1 = self.loss_ce(logits1, answer)
        
        target = F.softmax(logits1, dim=1)
        logits2 = F.log_softmax(logits2, dim=1)
        loss_KL2 = self.KL_loss(logits2, target)
        
        logits3 = F.log_softmax(logits3, dim=1)
        loss_KL3 = self.KL_loss(logits3, target)
        loss = loss_ce1 + loss_KL2 + loss_KL3
        
        # loss_ce1 = self.loss_ce(logits1, answer)
        # loss_ce2 = self.loss_ce(logits2, answer)
        # loss_ce3 = self.loss_ce(logits3, answer)
        # loss = loss_ce1 + loss_ce2 +loss_ce3
        
        
        if val:
            # score1 [4, 12, 20, 38]
            # score2 [4, 12, 20, 57]
            # attentions_mask [4, 12, 57]
            # fine_mask [4, 12, 57, 57]
            # score = torch.cat((score2[:, :, 0, :-20].unsqueeze(1), attentions_mask[:, :, :-20].unsqueeze(1), 
            #                    fine_mask[:, :, 0, :-20].unsqueeze(1)), dim=1)
            
            score = fine_mask
            return logits1, score
        
        return logits1, loss
    
    def create_mask(self, attn_score):
        values, indices = attn_score[:, :, 1:].topk(25, dim=2, largest=True)
        attn_mask = [[[1e-5 for _ in range(attn_score.shape[2])] for _ in range(attn_score.shape[1])] for _ in range(attn_score.shape[0])]
        attn_mask = torch.tensor(attn_mask).to(self.device)
        
        for i in range(len(indices)):
            for j in range(len(indices[0])):
                attn_mask[i, j, indices[i, j, :]] = 1
        return attn_mask 

class Bistep_vqa9(nn.Module):
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
        self.vit_encoder1, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.vit_encoder2, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size2, vit_grad_ckpt, vit_ckpt_layer)
        # self.sentence_bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)
        self.sentence_bert = SentenceTransformer('./pre_model/mpnet-base', device=self.device)
        # self.gru = nn.GRU(input_size=self.encoder_config.hidden_size, hidden_size=self.encoder_config.hidden_size, num_layers=2)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased')
        # modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel(config=modelConfig)
        
        # self.encoder1 = Encoder_Base(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        # self.encoder2 = Encoder_Base(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        # self.encoder3 = Encoder_Ques(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        
        # self.fusion = Fusion(self.encoder_config.hidden_size, \
        #     self.encoder_config.heads, self.encoder_config.hidden_dropout_prob)
        
        ##############
        # self.fusion1 = Text_base_attention5(self.encoder_config.hidden_size, self.encoder_config.heads, 
        #             self.encoder_config.hidden_dropout_prob)
        
        ################
        # self.attn_1 = MultiHeadAttention(self.encoder_config.heads, self.encoder_config.hidden_size, 
        #                                  dropout=self.encoder_config.hidden_dropout_prob)
        # self.ff4 = PositionWiseFeedForward(self.encoder_config.hidden_size)
        
        self.encoder1 = Base_Encoder(self.encoder_config.heads, self.encoder_config.hidden_size, 
                                     dropout=self.encoder_config.hidden_dropout_prob)
        self.encoder2 = Base_Encoder(self.encoder_config.heads, self.encoder_config.hidden_size, 
                                     dropout=self.encoder_config.hidden_dropout_prob)
        self.fusion1 = Text_base_attention7(self.encoder_config.hidden_size, self.encoder_config.heads, 
                    self.encoder_config.hidden_dropout_prob)
        # self.mfb = MFB(self.encoder_config, self.encoder_config.encoder_width, self.encoder_config.encoder_width, True)
        # self.ban = BAN(self.encoder_config)
        
        self.ff1 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, output_size))
        self.ff2 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, output_size))
        self.ff3 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
                                nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                nn.Linear(self.encoder_config.mlp_size, output_size))
        # self.classifier_open_close = nn.Sequential(nn.Linear(self.encoder_config.hidden_size, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        # self.softmax = F.softmax(dim=1)
        # self.log_softmax1 = F.log_softmax(dim=1)
        # self.log_softmax2 = F.log_softmax(dim=1)
        self.norm = nn.LayerNorm(self.encoder_config.hidden_size)
        self.loss_ce = nn.CrossEntropyLoss()
        # self.multi_label_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
        self.KL_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        
    def forward(self, image, tokens, segment_ids, input_mask, answer=None, 
                id2ans=None, ans_tokens_list=None, question=None, ques_target=None, val=None):
        
        bs = image.shape[0]     # 4
        nh = self.encoder_config.heads
        w_featmap=image.shape[-2] //self.encoder_config.patch_size    # 3
        h_featmap=image.shape[-1] //self.encoder_config.patch_size    # 3
        
        v_feat1 = self.vit_encoder1(image)  # [4, 10, 768]
        v_feat2 = self.vit_encoder2(image)  # [4, 10, 768]
        # v_feat2 = v_feat1
        w_feat = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        s_feat = torch.tensor(self.sentence_bert.encode(question, device=self.device)).to(self.device)
        # wo Strans
        # s_feat = w_feat[:, 0, :]
        ################
        # z_ws = self.attn_1(w_feat, s_feat.unsqueeze(1), s_feat.unsqueeze(1))#######
        # z_ws = self.ff4(z_ws)########
        
        # z_v1s, coarse_mask = self.encoder1(v_feat1, z_ws, return_mask=True)   ######3
        # z_v2w, fine_mask = self.encoder2(v_feat2, z_ws, return_mask=True) #######
        z_v1s, coarse_mask = self.encoder1(v_feat1, s_feat.unsqueeze(1), return_mask=True)   # [2, 12, 11, 11]
        # z_v1s, coarse_mask = self.encoder1(v_feat2, s_feat.unsqueeze(1), return_mask=True)
        z_v2w, fine_mask = self.encoder2(v_feat2, w_feat, return_mask=True) # [2, 12, 57, 57]
        
        
        # coarse_score = coarse_mask[:, :, 0, 1:-20].reshape(bs, nh, w_featmap, h_featmap)    ####
        coarse_score = coarse_mask[:, :, 0, 1:-1].reshape(bs, nh, w_featmap, h_featmap)
        upsample_size = self.encoder_config.patch_size//self.encoder_config.patch_size2
        attentions_mask = nn.functional.interpolate(coarse_score.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        attentions_mask = attentions_mask.reshape(bs, nh, -1)
        attentions_mask = torch.cat((coarse_mask[:, :, 0, 0].unsqueeze(-1), attentions_mask, 
                                     torch.ones(bs, nh, len(tokens[0])).to(self.device)), dim=-1)
        # attentions_mask = torch.cat((coarse_mask[:, :, 0, 0].unsqueeze(-1), attentions_mask, 
        #                              coarse_mask[:, :, 0, -1].unsqueeze(-1).repeat(1, 1, len(tokens[0]))), dim=-1)
        
        # a = z_v1s[:, 1:-20, :].transpose(1, 2)########3
        a = z_v1s[:, 1:-1, :].transpose(1, 2) # [bs, 51, 768]->[bs, 49, 768]->[bs, 768, 49]
        b = a.reshape(bs, self.encoder_config.hidden_size, w_featmap, h_featmap) # [bs, 768, 49]->[bs, 768, 7, 7]
        # [bs, 768, 7, 7] -> [bs, 768, 1, 7, 7]->[bs, 768, 2, 14, 14]->[bs, 768, 14, 14]
        c = nn.functional.interpolate(b.unsqueeze(2), scale_factor=upsample_size, mode="nearest")[:, :, 0, :, :]
        d = c.reshape(bs, self.encoder_config.hidden_size, -1).transpose(1, 2)  # [bs, 768, 14, 14]->[bs, 768, 196]->[bs, 196, 768]
        v_coarse = torch.cat((z_v1s[:, 0, :].unsqueeze(1), d, z_v1s[:, -1, :].unsqueeze(1)), dim=1)   # [bs, 196, 768]->[bs, 197, 768]
        
        ###########
        # attn_mask1 = fine_mask[:, :, 0, :v_coarse.shape[1]] - attentions_mask[:, :, :v_coarse.shape[1]]
        # attn_mask1 = data_normal_2d(attn_mask1, dim=-1, device=self.device)
        # attn_mask1 = torch.cat((attn_mask1, torch.ones(bs, nh, len(tokens[0])-1).to(self.device)), dim=-1)
        # output, z_ws = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask[:, :, :-19], attn_mask1)
        # output, z_ws = self.fusion1(z_v1s, z_v2w, w_feat, s_feat.unsqueeze(1))
        # output, z_ws = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask, fine_mask[:, :, 0, :-19])
        
        # output, z_ws = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask[:, :, :-19], fine_mask[:, :, 0, :])
        # output, z_ws = self.fusion1(v_coarse, z_v2w, z_ws, s_feat.unsqueeze(1), attentions_mask, fine_mask[:, :, 0, :])#########
        # output, z_ws = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask, fine_mask[:, :, 0, :])
        output, z_ws, score1, score2 = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask, fine_mask[:, :, 0, :])
        
        # attentions_mask = torch.cat((coarse_mask[:, :, 0, :], torch.ones(bs, nh, len(tokens[0])-1).to(self.device)), dim=-1)
        # output, z_ws, score1, score2 = self.fusion1(z_v1s, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask, fine_mask[:, :, 0, :])
        
        # BAN
        # output, z_ws = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask, fine_mask[:, :, 0, :])
        # output, att = self.ban(z_ws, output)    # att [4, 8, 57, 20]
        # MFB
        # output, z_ws = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1), attentions_mask, fine_mask[:, :, 0, :])
        # output, exp_out = self.mfb(output, z_ws[:, 0, :].unsqueeze(1))
        # wo coarse ViT
        # coarse_mask = torch.cat((coarse_mask[:, :, 0, :], torch.ones(bs, nh, len(tokens[0])-1).to(self.device)), dim=-1)
        # output, z_ws = self.fusion1(z_v1s, z_v2w, w_feat, s_feat.unsqueeze(1), coarse_mask, fine_mask[:, :, 0, :])
        # wo mask
        # output, z_ws = self.fusion1(v_coarse, z_v2w, w_feat, s_feat.unsqueeze(1))
        
        logits1 = self.ff1(output.mean(1))
        logits2 = self.ff2(z_v1s.mean(1))
        logits3 = self.ff3(z_v2w.mean(1))
        # logits1 = self.ff1(output[:, 0, :])
        # logits2 = self.ff2(z_v1s[:, 0, :])
        # logits3 = self.ff3(z_v2w[:, 0, :])
        # logits1 = self.ff1(output)
        # logits2 = self.ff2(z_v1s)
        # logits3 = self.ff3(z_v2w)
        
        # loss_ce1 = self.loss_ce(logits1, answer)
        # loss_ce2 = self.loss_ce(logits2, answer)
        # loss_ce3 = self.loss_ce(logits3, answer)
        # loss = loss_ce1 + loss_ce2 + loss_ce3
        
        loss_ce1 = self.loss_ce(logits1, answer)
        target = F.softmax(logits1, dim=1)
        logits2 = F.log_softmax(logits2, dim=1)
        loss_KL2 = self.KL_loss(logits2, target)
        
        logits3 = F.log_softmax(logits3, dim=1)
        loss_KL3 = self.KL_loss(logits3, target)
        
        loss = loss_ce1 + loss_KL2 + loss_KL3
        
        
        if val:
            score = torch.cat((score2[:, :, 0, :-20].unsqueeze(1), attentions_mask[:, :, :-20].unsqueeze(1), 
                               fine_mask[:, :, 0, :-20].unsqueeze(1)), dim=1)
            return logits1, score
        
        return logits1, loss
    
    def create_mask(self, attn_score):
        values, indices = attn_score[:, :, 1:].topk(25, dim=2, largest=True)
        attn_mask = [[[1e-5 for _ in range(attn_score.shape[2])] for _ in range(attn_score.shape[1])] for _ in range(attn_score.shape[0])]
        attn_mask = torch.tensor(attn_mask).to(self.device)
        
        for i in range(len(indices)):
            for j in range(len(indices[0])):
                attn_mask[i, j, indices[i, j, :]] = 1
        return attn_mask 


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

class Fusion(nn.Module):        
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
    
    def forward(self, z_ws, z_vs, z_vw, attn_mask):
        z_vw = self.norm_1(z_vw)
        z_vs = self.norm_2(z_vs)
        z_v = torch.add(z_vw, z_vs)
        z_vt, scores = self.attn_1(z_v, z_ws, z_ws, mask=attn_mask, return_attn=True)
        z_v = z_vw + self.norm_3(self.dropout_1(z_vt))
        output = z_ws + self.dropout_2(self.attn_2(z_ws, z_v, z_v))
        output = self.dropout_3(self.attn_3(output, output, output))
        
        output = self.ff1(output)
    
        return output

class Encoder_Ques(nn.Module):        
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
        
    def forward(self, w_feat, s_feat):
        z_vt = self.dropout_1(self.attn_1(w_feat, s_feat, s_feat))
        z_vt = self.ff1(z_vt)
        return z_vt
        
class Encoder_Base(nn.Module):        
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
        
    def forward(self, z_v, z_t, return_score=False):
        z_vt1, z_score1 = self.attn_1(z_v, z_t, z_t, return_attn=True)
        z_vt1 = self.dropout_1(z_vt1)
        z = self.norm_1(z_vt1)
        z_vt2, z_score2 = self.attn_1(z, z, z, return_attn=True)
        z_vt = z_vt1 + self.dropout_2(z_vt2)
        z_vt = self.ff1(z_vt)
        if return_score:
            z_score1 = z_score1.squeeze(-1) + z_score2[:, :, :, 0]
        
        return z_vt, z_score1 

class Globe_Transfor(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, s_feat, v_feat):
        z_vs = torch.cat((s_feat, v_feat), dim=1)   # [4, 198, 768]
        z_vs1, score_vs = self.attn_1(z_vs, z_vs, z_vs, return_attn=True)
        z_vs = z_vs + self.dropout_1(z_vs1)
        z_vs = self.norm_1(z_vs)
        z_vs = self.dropout_2(self.attn_2(z_vs, z_vs, z_vs))
        
        return z_vs, score_vs
        
class Local_Transfor(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
    def forward(self, w_feat, s_feat, v_feat1, v_feat2, attn_mask):
        z_ws = self.dropout_1(self.attn_1(w_feat, s_feat, s_feat))  # [4, 20, 768]
        z_vw, scores = self.attn_2(v_feat2, z_ws, z_ws, mask=attn_mask, return_attn=True)
        z_wv = self.norm_1(self.dropout_2(z_vw))+self.norm_2(v_feat1)
        z_wv = self.dropout_3(self.attn_3(z_wv, z_wv, z_wv))
        
        return z_wv, z_ws

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)
        
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


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
        
    def forward(self, t_feat, v_feat, attn_mask=None, return_mask=False):
        z_t = self.norm_1(t_feat)
        z_t, score = self.attn_2(z_t, v_feat, v_feat, mask=attn_mask, return_mask=True)
        t_feat = t_feat + self.dropout_1(z_t)
        # t_feat = t_feat + self.dropout_1(self.attn_2(z_t, v_feat, v_feat, mask=attn_mask))
        z_t = self.norm_2(t_feat)
        t_feat = t_feat + self.dropout_2(self.attn_2(z_t, z_t, z_t))
        t_feat = self.ff2(t_feat)
        
        return t_feat, score

class Text_base_attention5(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
        
        self.encoder1 = Base_Encoder(heads, d_model, dropout=dropout)
        self.encoder2 = Base_Encoder(heads, d_model, dropout=dropout)
        
        self.cross_encoder1 = Cross_Encoder(heads, d_model, dropout=dropout)
        self.cross_encoder2 = Cross_Encoder(heads, d_model, dropout=dropout)
    
    def forward(self, v_feat1, v_feat2, w_feat, s_feat):
        z_ws = self.attn_1(w_feat, s_feat, s_feat)
        z_ws = self.ff1(z_ws)
        
        z_v1s, coarse_mask = self.encoder1(v_feat1, s_feat, return_mask=True)   # [2, 12, 11, 11]
        z_v2w = self.encoder2(v_feat2, w_feat)
        
        z_f1 = self.cross_encoder1(z_ws, z_v1s)
        z_f2 = self.cross_encoder1(z_f1, z_v2w)
        # output = self.norm_1(z_f2)
        # output = self.dropout_1(self.ff1(output))
        return z_f2, z_ws, z_v1s, z_v2w
    
class Text_base_attention6(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1 = PositionWiseFeedForward(d_model)
        
        self.cross_encoder1 = Cross_Encoder(heads, d_model, dropout=dropout)
        self.cross_encoder2 = Cross_Encoder(heads, d_model, dropout=dropout)
    
    def forward(self, z_v1s, z_v2w, w_feat, s_feat, coarse_mask=None, fine_mask=None, return_mask=False):
        z_ws = self.attn_1(w_feat, s_feat, s_feat)
        z_ws = self.ff1(z_ws+w_feat)

        # z_v1s = self.norm_1(z_v1s)
        # z_v2w = self.norm_2(z_v2w)
        # z_v1s[:, :-1, :] += z_v2w[:, :-20, :]
        # z_f1 = self.cross_encoder1(z_ws, z_v1s, attn_mask=coarse_mask)
        # z_f2 = self.cross_encoder1(z_f1, z_v2w, attn_mask=fine_mask)
        ########3
        z_f1, score1 = self.cross_encoder1(z_ws, z_v1s, attn_mask=fine_mask, return_mask=return_mask)
        z_f2, score2 = self.cross_encoder1(z_f1, z_v2w, attn_mask=coarse_mask, return_mask=return_mask)
        
        #####3
        # z_f1 = self.cross_encoder1(z_ws, z_v2w, attn_mask=coarse_mask)
        # z_f2 = self.cross_encoder1(z_f1, z_v1s, attn_mask=fine_mask)
        # output = self.norm_1(z_f2)
        # output = self.dropout_1(self.ff1(output))
        if return_mask:
            return z_f2, z_ws, score1, score2
        
        return z_f2, z_ws

class Text_base_attention7(nn.Module):
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
        self.cross_encoder2 = Cross_Encoder(heads, d_model, dropout=dropout)
    
    def forward(self, z_v1s, z_v2w, w_feat, s_feat, coarse_mask=None, fine_mask=None):
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
        z_f2, score2 = self.cross_encoder1(z_f1, z_v2w, attn_mask=fine_mask)
        
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