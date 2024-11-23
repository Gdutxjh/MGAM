import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertTokenizer, BertModel
from torchvision import models as torch_models
import math
import numpy as np
import copy

from models.vit import VisionTransformer, interpolate_pos_embed
from models.resnet import resnet152 
from models.med import BertConfig, BertLMHeadModel
from models.multi_attn import MultiHeadAttention
from models.CoAttn import MFB, CoAtt
from models.gumbel_softmax import gumbel_softmax
from models.lxmert_model import LXRTFeatureExtraction as VisualBertForLXRFeature
from models.lxmert_model import LXRTModel, LXRTEncoder
from functools import partial
from sentence_transformers import SentenceTransformer

class BiVision_VQA3(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0"
                 ):
        super().__init__()

        self.encoder_config = BertConfig.from_json_file(med_config)
        # self.vae1 = VAE(input_dim=self.encoder_config.hidden_size)
        # self.vae2 = VAE(input_dim=self.encoder_config.hidden_size)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/pytorch_model.bin', config=modelConfig)

        # self.lxmert = VisualBertForLXRFeature.from_pretrained(
        #     "bert-base-uncased",
        #     mode='x'
        # )
        # self.encoder = self.lxmert.bert
        
        self.local_encoder = LXRTEncoder(modelConfig)
        self.gobal_encoder = LXRTEncoder(modelConfig)
        
        self.encoder_config.encoder_width = vision_width
        # self.v_att_proj1 = nn.Linear(768, 2048)
        # self.gobal_mfb = MFB(self.encoder_config, self.encoder_config.encoder_width, self.encoder_config.encoder_width, True)
        
        self.v_att_proj1 = nn.Linear(768, 1024)
        self.v_att_proj2 = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
        
        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            # hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
            hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)
        # self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
        #     hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        # self.local_mfb = MFB()
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)

        self.fusion = mlp(in_features=1024*2, 
                          hidden_features=self.encoder_config.mlp_size, 
                          out_features=output_size, 
                          act_layer=nn.ReLU, 
                          drop=self.encoder_config.hidden_dropout_prob)
        
        # self.vae_mlp = mlp(in_features=self.encoder_config.hidden_size, 
        #                   hidden_features=self.encoder_config.mlp_size, 
        #                   out_features=self.encoder_config.hidden_size, 
        #                   act_layer=nn.ReLU, 
        #                   drop=self.encoder_config.hidden_dropout_prob)

    def forward(self, image, tokens, segment_ids, input_mask, caption="", train=False):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768
        
        image_global_embeds = self.resnet_encoder(image)  # batch_size*768
        
        # image_global_embeds = image_local_embeds[:, 0, :]
        image_global_embeds = torch.cat((image_global_embeds.unsqueeze(1), image_local_embeds[:, 0, :].unsqueeze(1)), dim=1)
        image_local_embeds = image_local_embeds[:, 1:, :]

        # caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=self.encoder_config.seq_max_len, 
        #                           return_tensors="pt").to(image.device) 
        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        
        # local_output = self.local_attention(question_embeds, image_local_embeds, image_local_embeds)   # bs*seq_len*768
        # gobal_output = self.gobal_attention(question_embeds, image_global_embeds, image_global_embeds) # bs*seq_len*768
        # seman_output = self.semantics_attention(question, caption, caption)     # bs*seq_len*768
        
        # feat_seq, pooled_output = self.encoder(tokens, segment_ids, input_mask,
        #                                     visual_feats=image_local_embeds,
        #                                     visual_attention_mask=None)
        # question_embeds, image_local_embeds = feat_seq
        question_embeds, image_local_embeds = self.local_encoder(question_embeds, input_mask, image_local_embeds)
        question_embeds, image_global_embeds = self.local_encoder(question_embeds, input_mask, image_global_embeds)
        
        image_local_embeds = self.v_att_proj1(image_local_embeds)
        image_global_embeds = self.v_att_proj2(image_global_embeds)
        question_embeds = self.l_att_proj(question_embeds)
        
        sim_matrix_v2l1 = torch.matmul(image_local_embeds, question_embeds.transpose(1,2))  # b * v_length * l_length
        kg_output1, k1 = torch.topk(sim_matrix_v2l1, dim=-1, k=1)
        hard_attention_value1 = gumbel_softmax(kg_output1.squeeze())
        head1 = (image_local_embeds * hard_attention_value1.unsqueeze(-1)).sum(-2)
        
        sim_matrix_v2l2 = torch.matmul(image_global_embeds, question_embeds.transpose(1,2))  # b * v_length * l_length
        kg_output2, k = torch.topk(sim_matrix_v2l2, dim=-1, k=1)
        hard_attention_value2 = gumbel_softmax(kg_output2.squeeze())
        head2 = (image_global_embeds * hard_attention_value2.unsqueeze(-1)).sum(-2)
        
        # local_hat, local_mu, local_log_var_vae = self.vae1(local_output.squeeze(1))
        # gobal_hat, gobal_mu, gobal_log_var_vae = self.vae2(gobal_output.squeeze(1))
        # h = torch.cat(local_output.squeeze(1), gobal_output.squeeze(1), seman_output.squeeze(1), dim=1)
        # h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1), gobal_output.squeeze(1)), dim=1)
        # h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1)), dim=1)
        h = torch.cat((head1, head2), dim=1)
        # local_hat, _ = self.vae_mlp(local_hat)
        # gobal_hat, _ = self.vae_mlp(gobal_hat)
        logits, _ = self.fusion(h)
        # logits = F.softmax(output, dim=-1) # bs*ans_size
        # if train:
        #     return logits, local_hat, local_mu, local_log_var_vae, gobal_hat, gobal_mu, gobal_log_var_vae
        
        return logits 
    
    def l2_norm(self, input, dim=-1):
        norm = torch.norm(input, dim=dim, keepdim=True)
        output = torch.div(input, norm)
        return output
    
class BiVision_VQA16(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0"
                 ):
        super().__init__()

        self.encoder_config = BertConfig.from_json_file(med_config)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)

        self.encoder_config.encoder_width = vision_width

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 

        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        # self.local_mfb = MFB()
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)

        self.fusion = mlp(in_features=self.encoder_config.hidden_size*3, 
                          hidden_features=self.encoder_config.mlp_size, 
                          out_features=output_size, 
                          act_layer=nn.ReLU, 
                          drop=self.encoder_config.hidden_dropout_prob)

    def forward(self, image, tokens, segment_ids, input_mask, caption="", train=True):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768
        image_global_embeds = self.resnet_encoder(image)  # batch_size*768
        image_global_embeds = image_global_embeds.unsqueeze(1)

        # caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=self.encoder_config.seq_max_len, 
        #                           return_tensors="pt").to(image.device) 
        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        
        local_output = self.local_attention(question_embeds, image_local_embeds, image_local_embeds)   # bs*seq_len*768
        gobal_output = self.gobal_attention(question_embeds, image_global_embeds, image_global_embeds) # bs*seq_len*768
        # seman_output = self.semantics_attention(question, caption, caption)     # bs*seq_len*768
        
        # h = torch.cat(local_output.squeeze(1), gobal_output.squeeze(1), seman_output.squeeze(1), dim=1)
        h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1), gobal_output.squeeze(1)), dim=1)
        output, hid_emb = self.fusion(h)
        # logits = F.softmax(output, dim=-1) # bs*ans_size
        # logits = self.l2_norm(output, dim=-1)
        hid_emb = self.l2_norm(hid_emb, dim=-1)
        return output, hid_emb 
    
    def l2_norm(self, input, dim=-1):
        norm = torch.norm(input, dim=dim, keepdim=True)
        output = torch.div(input, norm)
        return output

class BiVision_VQA6(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0"
                 ):
        super().__init__()

        self.encoder_config = BertConfig.from_json_file(med_config)
        # self.vae1 = VAE(input_dim=self.encoder_config.hidden_size)
        # self.vae2 = VAE(input_dim=self.encoder_config.hidden_size)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 

        # self.lxmert = VisualBertForLXRFeature.from_pretrained(
        #     "bert-base-uncased",
        #     mode='x'
        # )
        # self.encoder = self.lxmert.bert
        
        self.encoder = LXRTEncoder(modelConfig)
        
        self.encoder_config.encoder_width = vision_width
        # self.v_att_proj1 = nn.Linear(768, 2048)
        
        self.v_att_proj = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
        
        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        # self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
        #     hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        # self.local_mfb = MFB()
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)

        self.fusion = mlp(in_features=1024, 
                          hidden_features=self.encoder_config.mlp_size, 
                          out_features=output_size, 
                          act_layer=nn.ReLU, 
                          drop=self.encoder_config.hidden_dropout_prob)
        
        # self.vae_mlp = mlp(in_features=self.encoder_config.hidden_size, 
        #                   hidden_features=self.encoder_config.mlp_size, 
        #                   out_features=self.encoder_config.hidden_size, 
        #                   act_layer=nn.ReLU, 
        #                   drop=self.encoder_config.hidden_dropout_prob)

    def forward(self, image, tokens, segment_ids, input_mask, caption="", train=False):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768
        
        # image_global_embeds = self.resnet_encoder(image)  # batch_size*768
        # image_global_embeds = image_global_embeds.unsqueeze(1)
        
        # image_global_embeds = image_local_embeds[:, 0, :]
        # image_local_embeds = image_local_embeds[:, 1:, :]

        # caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=self.encoder_config.seq_max_len, 
        #                           return_tensors="pt").to(image.device) 
        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        
        # local_output = self.local_attention(question_embeds, image_local_embeds, image_local_embeds)   # bs*seq_len*768
        # gobal_output = self.gobal_attention(question_embeds, image_global_embeds, image_global_embeds) # bs*seq_len*768
        # seman_output = self.semantics_attention(question, caption, caption)     # bs*seq_len*768
        
        # feat_seq, pooled_output = self.encoder(tokens, segment_ids, input_mask,
        #                                     visual_feats=image_local_embeds,
        #                                     visual_attention_mask=None)
        # question_embeds, image_local_embeds = feat_seq
        question_embeds, image_local_embeds = self.encoder(question_embeds, input_mask, image_local_embeds, )
        
        image_local_embeds = self.v_att_proj(image_local_embeds)
        question_embeds = self.l_att_proj(question_embeds)
        
        sim_matrix_v2l = torch.matmul(image_local_embeds, question_embeds.transpose(1,2))  # b * v_length * l_length
        kg_output, k = torch.topk(sim_matrix_v2l, dim=-1, k=1)
        hard_attention_value = gumbel_softmax(kg_output.squeeze())
        head = (image_local_embeds * hard_attention_value.unsqueeze(-1)).sum(-2)
        
        # local_hat, local_mu, local_log_var_vae = self.vae1(local_output.squeeze(1))
        # gobal_hat, gobal_mu, gobal_log_var_vae = self.vae2(gobal_output.squeeze(1))
        # h = torch.cat(local_output.squeeze(1), gobal_output.squeeze(1), seman_output.squeeze(1), dim=1)
        # h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1), gobal_output.squeeze(1)), dim=1)
        # h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1)), dim=1)
        # h = torch.cat((local_hat, gobal_hat), dim=1)
        # local_hat, _ = self.vae_mlp(local_hat)
        # gobal_hat, _ = self.vae_mlp(gobal_hat)
        logits, _ = self.fusion(head)
        # logits = F.softmax(output, dim=-1) # bs*ans_size
        # if train:
        #     return logits, local_hat, local_mu, local_log_var_vae, gobal_hat, gobal_mu, gobal_log_var_vae
        
        return logits 

class BiVision_VQA7(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0"
                 ):
        super().__init__()

        self.encoder_config = BertConfig.from_json_file(med_config)
        # self.vae1 = VAE(input_dim=self.encoder_config.hidden_size)
        # self.vae2 = VAE(input_dim=self.encoder_config.hidden_size)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 

        
        self.local_encoder = LXRTEncoder(modelConfig)
        self.gobal_encoder = LXRTEncoder(modelConfig)
        
        self.encoder_config.encoder_width = vision_width
        # self.v_att_proj1 = nn.Linear(768, 2048)
        
        self.v_att_proj1 = nn.Linear(768, 1024)
        self.v_att_proj2 = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)
        
        self.mfh_proj = nn.Linear(2048, 1024)

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
        self.CoAtt = CoAtt(self.encoder_config)
        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        # self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
        #     hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        # self.local_mfb = MFB()
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)

        self.fusion = mlp(in_features=1024*2, 
                          hidden_features=self.encoder_config.mlp_size, 
                          out_features=output_size, 
                          act_layer=nn.ReLU, 
                          drop=self.encoder_config.hidden_dropout_prob)
        
        # self.vae_mlp = mlp(in_features=self.encoder_config.hidden_size, 
        #                   hidden_features=self.encoder_config.mlp_size, 
        #                   out_features=self.encoder_config.hidden_size, 
        #                   act_layer=nn.ReLU, 
        #                   drop=self.encoder_config.hidden_dropout_prob)

    def forward(self, image, tokens, segment_ids, input_mask, caption="", train=False):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768
        
        image_global_embeds = self.resnet_encoder(image)  # batch_size*768
        
        # image_global_embeds = image_local_embeds[:, 0, :]
        image_global_embeds = torch.cat((image_global_embeds.unsqueeze(1), image_local_embeds[:, 0, :].unsqueeze(1)), dim=1)
        image_local_embeds = image_local_embeds[:, 1:, :]

        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        
        # local_output = self.local_attention(question_embeds, image_local_embeds, image_local_embeds)   # bs*seq_len*768
        # gobal_output = self.gobal_attention(question_embeds, image_global_embeds, image_global_embeds) # bs*seq_len*768
        # seman_output = self.semantics_attention(question, caption, caption)     # bs*seq_len*768
        
        # feat_seq, pooled_output = self.encoder(tokens, segment_ids, input_mask,
        #                                     visual_feats=image_local_embeds,
        #                                     visual_attention_mask=None)
        # question_embeds, image_local_embeds = feat_seq
        question_embeds, image_local_embeds = self.local_encoder(question_embeds, input_mask, image_local_embeds)
        question_embeds, image_global_embeds = self.local_encoder(question_embeds, input_mask, image_global_embeds)
        
        image_local_embeds = self.v_att_proj1(image_local_embeds)
        image_global_embeds = self.v_att_proj2(image_global_embeds)
        question_embeds = self.l_att_proj(question_embeds)
        
        sim_matrix_v2l1 = torch.matmul(image_local_embeds, question_embeds.transpose(1,2))  # b * v_length * l_length
        kg_output1, k1 = torch.topk(sim_matrix_v2l1, dim=-1, k=1)
        hard_attention_value1 = gumbel_softmax(kg_output1.squeeze())
        head1 = (image_local_embeds * hard_attention_value1.unsqueeze(-1)).sum(-2)
        
        # sim_matrix_v2l2 = torch.matmul(image_global_embeds, question_embeds.transpose(1,2))  # b * v_length * l_length
        # kg_output2, k = torch.topk(sim_matrix_v2l2, dim=-1, k=1)
        # hard_attention_value2 = gumbel_softmax(kg_output2.squeeze())
        # head2 = (image_global_embeds * hard_attention_value2.unsqueeze(-1)).sum(-2)
        head2 = self.CoAtt(image_global_embeds, question_embeds)
        head2 = self.mfh_proj(head2)
        
        # local_hat, local_mu, local_log_var_vae = self.vae1(local_output.squeeze(1))
        # gobal_hat, gobal_mu, gobal_log_var_vae = self.vae2(gobal_output.squeeze(1))
        # h = torch.cat(local_output.squeeze(1), gobal_output.squeeze(1), seman_output.squeeze(1), dim=1)
        # h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1), gobal_output.squeeze(1)), dim=1)
        # h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1)), dim=1)
        
        
        h = torch.cat((head1, head2), dim=1)
        # local_hat, _ = self.vae_mlp(local_hat)
        # gobal_hat, _ = self.vae_mlp(gobal_hat)
        logits, _ = self.fusion(h)
        # logits = F.softmax(output, dim=-1) # bs*ans_size
        # if train:
        #     return logits, local_hat, local_mu, local_log_var_vae, gobal_hat, gobal_mu, gobal_log_var_vae
        
        return logits 

class BiVision_VQA8(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0"
                 ):
        super().__init__()

        self.encoder_config = BertConfig.from_json_file(med_config)
        # self.vae1 = VAE(input_dim=self.encoder_config.hidden_size)
        # self.vae2 = VAE(input_dim=self.encoder_config.hidden_size)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)
        # self.resnet = Transfer(self.encoder_config)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 

        
        self.local_encoder = LXRTEncoder(modelConfig)
        # self.gobal_encoder = LXRTEncoder(modelConfig)
        
        self.encoder_config.encoder_width = vision_width
        # self.v_att_proj1 = nn.Linear(768, 2048)
        
        self.v_att_proj1 = nn.Linear(768, 1024)
        self.v_att_proj2 = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)
        
        # self.mfh_proj = nn.Linear(2048, 1024)

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
         #self.CoAtt = CoAtt(self.encoder_config)
        # self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
        #     hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.MFB_size, h=2, \
            hidden_size=self.encoder_config.MFB_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        # self.local_mfb = MFB(self.encoder_config, self.encoder_config.MFB_size, self.encoder_config.MFB_size)
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)

        # self.fusion = mlp(in_features=1024*2, 
        #                   hidden_features=self.encoder_config.mlp_size, 
        #                   out_features=output_size, 
        #                   act_layer=nn.ReLU, 
        #                   drop=self.encoder_config.hidden_dropout_prob)
        
        # self.classifier = nn.Sequential(nn.Linear(1024, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        self.classifier = nn.Sequential(nn.Linear(1024, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        # self.vae_mlp = mlp(in_features=self.encoder_config.hidden_size, 
        #                   hidden_features=self.encoder_config.mlp_size, 
        #                   out_features=self.encoder_config.hidden_size, 
        #                   act_layer=nn.ReLU, 
        #                   drop=self.encoder_config.hidden_dropout_prob)

    def forward(self, image, tokens, segment_ids, input_mask, train=False, not_pretrain=True):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768
        
        # image_global_embeds = self.resnet(image)  # batch_size*768
        
        # image_global_embeds = image_local_embeds[:, 0, :]
        # image_global_embeds = torch.cat((image_global_embeds.unsqueeze(1), image_local_embeds[:, 0, :].unsqueeze(1)), dim=1)
        # image_local_embeds = image_local_embeds[:, 1:, :]

        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        
        # local_output = self.local_attention(question_embeds, image_local_embeds, image_local_embeds)   # bs*seq_len*768
        # gobal_output = self.gobal_attention(question_embeds, image_global_embeds, image_global_embeds) # bs*seq_len*768
        # seman_output = self.semantics_attention(question, caption, caption)     # bs*seq_len*768
        
        # feat_seq, pooled_output = self.encoder(tokens, segment_ids, input_mask,
        #                                     visual_feats=image_local_embeds,
        #                                     visual_attention_mask=None)
        # question_embeds, image_local_embeds = feat_seq
        question_embeds, image_local_embeds = self.local_encoder(question_embeds, input_mask, image_local_embeds)
        # question_embeds, image_global_embeds = self.local_encoder(question_embeds, input_mask, image_global_embeds)
        
        image_local_embeds = self.v_att_proj1(image_local_embeds)
        # image_global_embeds = self.v_att_proj2(image_global_embeds)
        question_embeds = self.l_att_proj(question_embeds)
        
        sim_matrix_v2l1 = torch.matmul(image_local_embeds, question_embeds.transpose(1,2))  # bs * 362 * 20
        kg_output1, k1 = torch.topk(sim_matrix_v2l1, dim=-1, k=1)           # bs * 362 * 1
        hard_attention_value1 = gumbel_softmax(kg_output1.squeeze())        # bs * 362
        head1 = (image_local_embeds * hard_attention_value1.unsqueeze(-1)).sum(-2)  # bs  1024
        head = self.gobal_attention(question_embeds, head1.unsqueeze(1), head1.unsqueeze(1))
        
        # sim_matrix_v2l2 = torch.matmul(image_global_embeds, question_embeds.transpose(1,2))  # b * v_length * l_length
        # kg_output2, k = torch.topk(sim_matrix_v2l2, dim=-1, k=1)
        # hard_attention_value2 = gumbel_softmax(kg_output2.squeeze())
        # head2 = (image_global_embeds * hard_attention_value2.unsqueeze(-1)).sum(-2)
        # head2 = self.CoAtt(image_global_embeds, question_embeds)
        # head2 = self.mfh_proj(head2)
        
        # local_hat, local_mu, local_log_var_vae = self.vae1(local_output.squeeze(1))
        # gobal_hat, gobal_mu, gobal_log_var_vae = self.vae2(gobal_output.squeeze(1))
        # h = torch.cat(local_output.squeeze(1), gobal_output.squeeze(1), seman_output.squeeze(1), dim=1)
        # h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1), gobal_output.squeeze(1)), dim=1)
        # h = torch.cat((question_embeds[:, 0, :], local_output.squeeze(1)), dim=1)
        # head, _ = self.local_mfb(head1.unsqueeze(1), question_embeds)
        
        # head = self.gobal_attention(head, image_global_embeds, image_global_embeds)
        # h = torch.cat((head1, head2), dim=1)
        # local_hat, _ = self.vae_mlp(local_hat)
        # gobal_hat, _ = self.vae_mlp(gobal_hat)
        # logits, _ = self.fusion(h)
        # logits = F.softmax(output, dim=-1) # bs*ans_size
        # if train:
        #     return logits, local_hat, local_mu, local_log_var_vae, gobal_hat, gobal_mu, gobal_log_var_vae
        if not_pretrain:
            head = head.mean(1)
        logits = self.classifier(head)
        return logits

class BiVision_VQA9(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0",
                 batch_size = 16,
                 ):
        super().__init__()

        self.encoder_config = BertConfig.from_json_file(med_config)
        # self.vae1 = VAE(input_dim=self.encoder_config.hidden_size)
        # self.vae2 = VAE(input_dim=self.encoder_config.hidden_size)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)
        # self.resnet = Transfer(self.encoder_config)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 

        
        self.local_encoder = LXRTEncoder(modelConfig)
        # self.gobal_encoder = LXRTEncoder(modelConfig)
        
        self.encoder_config.encoder_width = vision_width
        # self.v_att_proj1 = nn.Linear(768, 2048)
        
        self.v_att_proj1 = nn.Linear(768, 1024)
        self.v_att_proj2 = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)
        
        # self.mfh_proj = nn.Linear(2048, 1024)
        self.temp = nn.Parameter(0.07*torch.ones([])) 

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
         #self.CoAtt = CoAtt(self.encoder_config)
        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.MFB_size, h=2, \
            hidden_size=self.encoder_config.MFB_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.local_mfb = MFB(self.encoder_config, self.encoder_config.MFB_size, self.encoder_config.MFB_size)
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)
        
        self.classifier1 = nn.Sequential(nn.Linear(1024, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        self.classifier2 = nn.Sequential(nn.Linear(1024, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        

    def forward(self, image, tokens, segment_ids, input_mask, answer_type="OPEN", alpha=0.2, training=False, not_pretrain=True):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768

        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        question_embeds, image_local_embeds = self.local_encoder(question_embeds, input_mask, image_local_embeds)
        
        #########
        # if training:
        #     image_feat = F.normalize(image_local_embeds[:, 0, :], dim=-1)
        #     text_feat = F.normalize(question_embeds[:, 0, :], dim=-1)
        #     sim_i2t_m = image_feat @ text_feat.t() / self.temp  
        #     sim_t2i_m = text_feat @ image_feat.t() / self.temp
            
        #     sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
        #     sim_targets.fill_diagonal_(1) 
        
        #     sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
        #     sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
        #     loss_i2t = -torch.sum(F.log_softmax(sim_i2t_m, dim=1)*sim_i2t_targets,dim=1).mean()
        #     loss_t2i = -torch.sum(F.log_softmax(sim_t2i_m, dim=1)*sim_t2i_targets,dim=1).mean() 
        #     loss_ita1 = (loss_i2t+loss_t2i)/2
        
        # question_embeds, image_global_embeds = self.local_encoder(question_embeds, input_mask, image_global_embeds)
        
        image_local_embeds = self.v_att_proj1(image_local_embeds)
        # image_global_embeds = self.v_att_proj2(image_global_embeds)
        question_embeds = self.l_att_proj(question_embeds)
        
        sim_matrix_v2l1 = torch.matmul(image_local_embeds, question_embeds.transpose(1,2))  # bs * 362 * 20
        kg_output1, k1 = torch.topk(sim_matrix_v2l1, dim=-1, k=1)           # bs * 362 * 1
        hard_attention_value1 = gumbel_softmax(kg_output1.squeeze())        # bs * 362
        head1 = (image_local_embeds * hard_attention_value1.unsqueeze(-1)).sum(-2)  # bs  1024
        
        head = self.gobal_attention(question_embeds, head1.unsqueeze(1), head1.unsqueeze(1))
        
        # if training:
        #     image_feat = F.normalize(head1, dim=-1)
        #     text_feat = F.normalize(question_embeds[:, 0, :], dim=-1)
        #     sim_i2t_m = image_feat @ text_feat.t() / self.temp  
        #     sim_t2i_m = text_feat @ image_feat.t() / self.temp
            
        #     sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
        #     sim_targets.fill_diagonal_(1) 
        
        #     sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
        #     sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
        #     loss_i2t = -torch.sum(F.log_softmax(sim_i2t_m, dim=1)*sim_i2t_targets,dim=1).mean()
        #     loss_t2i = -torch.sum(F.log_softmax(sim_t2i_m, dim=1)*sim_t2i_targets,dim=1).mean() 
        #     loss_ita2 = (loss_i2t+loss_t2i)/2
            
        #     loss_ita = (loss_ita1+loss_ita2)/2
        
        head = head.mean(1)
        # res = torch.tensor([])
        for i in range(len(answer_type)):
            if answer_type[i] == "OPEN":
                logits = self.classifier1(head[i])
            else:
                logits = self.classifier2(head[i])
            if i == 0:
                res = logits.unsqueeze(0)
            else:
                res = torch.cat((res, logits.unsqueeze(0)), dim=0)
        # logits = self.classifier1(head)
        # if training:
        #     return res, loss_ita
        return res

class BiVision_VQA10(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0",
                 batch_size = 16,
                 ):
        super().__init__()

        self.encoder_config = BertConfig.from_json_file(med_config)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)
        # self.resnet = Transfer(self.encoder_config)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 

        
        # self.local_encoder = LXRTEncoder(modelConfig)
        self.transformers = Transformer(self.encoder_config)
        # self.gobal_encoder = LXRTEncoder(modelConfig)
        
        self.encoder_config.encoder_width = vision_width
        # self.v_att_proj1 = nn.Linear(768, 2048)
        
        self.v_att_proj1 = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.hidden_size2)
        self.v_att_proj2 = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.hidden_size2)
        self.l_att_proj = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.hidden_size2)
        
        # self.mfh_proj = nn.Linear(2048, 1024)

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
         #self.CoAtt = CoAtt(self.encoder_config)
        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.MFB_size, h=2, \
            hidden_size=self.encoder_config.MFB_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        # self.local_mfb = MFB(self.encoder_config, self.encoder_config.MFB_size, self.encoder_config.MFB_size)
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)
        
        self.classifier1 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size2, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        self.classifier2 = nn.Sequential(nn.Linear(self.encoder_config.hidden_size2, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.temp = nn.Parameter(0.07*torch.ones([])) 
        self.queue_size = self.encoder_config.queue_size
        self.momentum = self.encoder_config.momentum
        self.itm_head = nn.Linear(self.encoder_config.hidden_size2, 2) 
        
        # create momentum models
        self.visual_encoder_m , _ = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer) 
        self.vision_proj_m = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.hidden_size2)
        self.text_encoder_m = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig)      
        self.text_proj_m = nn.Linear(self.encoder_config.hidden_size, self.encoder_config.hidden_size2)    
        
        self.model_pairs = [[self.vit_encoder,self.visual_encoder_m],
                            [self.v_att_proj1,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.l_att_proj,self.text_proj_m],
                           ]
        self.copy_params()
        
        # create the queue
        self.register_buffer("image_queue", torch.randn(self.encoder_config.hidden_size2, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.encoder_config.hidden_size2, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
        
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        

    def forward(self, image, tokens, segment_ids, input_mask, tokens_m, segment_ids_m, input_mask_m, alpha=0.2, training=True):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768

        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        # question_embeds, image_local_embeds = self.local_encoder(question_embeds, input_mask, image_local_embeds)
        
        h, attn_score = self.transformers(image_local_embeds, question_embeds, input_mask)
        h = h.mean(0)
        image_local_embeds, question_embeds = h[:, 1:38,:], h[:, 38:,:]
        question_embeds = torch.cat((h[:, 0, :].unsqueeze(1), question_embeds), dim=1)
        
        image_feat = F.normalize(self.v_att_proj1(image_local_embeds[:,0,:]),dim=-1)
        text_feat = F.normalize(self.l_att_proj(question_embeds[:,0,:]),dim=-1)  
        #########
        if training:
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image)
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
                image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)
                text_output_m = self.text_encoder_m(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
                text_feat_m = F.normalize(self.text_proj_m(text_output_m[:,0,:]),dim=-1)
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
        
        # question_embeds, image_global_embeds = self.local_encoder(question_embeds, input_mask, image_global_embeds)
        
        image_embeds = self.v_att_proj1(image_local_embeds)
        # # image_global_embeds = self.v_att_proj2(image_global_embeds)
        question_embeds = self.l_att_proj(question_embeds)
        head = self.CMF(image_embeds, question_embeds)
        
        # ITM
        if training:
            with torch.no_grad():
                bs = image.size(0)          
                weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
                weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
    
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
                
                # select a negative image for each text
            image_embeds_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

            # select a negative text for each image
            text_embeds_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_embeds_neg.append(question_embeds[neg_idx])
                text_atts_neg.append(input_mask[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
            text_atts_neg = torch.stack(text_atts_neg,dim=0) 
            
            text_embeds_all = torch.cat([question_embeds, text_embeds_neg],dim=0)        
            image_embeds_all = torch.cat([image_embeds_neg, image_embeds],dim=0)
            output_neg = self.CMF(image_embeds_all, text_embeds_all)
            
            vl_embeddings = torch.cat([head[:,0,:], output_neg[:,0,:]],dim=0)
            vl_output = self.itm_head(vl_embeddings)            

            itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                                dim=0).to(image.device)
            loss_itm = F.cross_entropy(vl_output, itm_labels) 
        
        ##================= MLM ========================##
        question_embeds_mask = self.text_encoder(input_ids=tokens_m, token_type_ids=segment_ids_m, position_ids=input_mask_m)[0]
        # question_embeds_mask, image_local_embeds_mask = self.local_encoder(question_embeds_mask, input_mask_m, image_local_embeds)
        h, attn_score = self.transformers(image_local_embeds, question_embeds_mask, input_mask)
        h = h.mean(0)
        # h = h[-1]
        image_local_embeds_mask, question_embeds_mask = h[:, 1:38,:], h[:, 38:,:]
        question_embeds_mask = torch.cat((h[:, 0, :].unsqueeze(1), question_embeds_mask), dim=1)
        
        image_feat_mask = self.v_att_proj1(image_local_embeds_mask)
        text_feat_mask = self.l_att_proj(question_embeds_mask)
        head_mask = self.CMF(image_feat_mask, text_feat_mask)
        # res = torch.tensor([])
        head_mask = head_mask.mean(1)
        logits = self.classifier1(head_mask)
        # for i in range(len(answer_type)):
        #     if answer_type[i] == "OPEN":
        #         logits = self.classifier1(head[i])
        #     else:
        #         logits = self.classifier2(head[i])
        #     if i == 0:
        #         res = logits.unsqueeze(0)
        #     else:
        #         res = torch.cat((res, logits.unsqueeze(0)), dim=0)
        # logits = self.classifier1(head)
        if training:
            return logits, loss_ita, loss_itm
        return logits
    
    def CMF(self, image_local_embeds, question_embeds):
        sim_matrix_v2l1 = torch.matmul(image_local_embeds, question_embeds.transpose(1,2))  # bs * 362 * 20
        kg_output1, k1 = torch.topk(sim_matrix_v2l1, dim=-1, k=1)           # bs * 362 * 1
        hard_attention_value1 = gumbel_softmax(kg_output1.squeeze())        # bs * 362
        head1 = (image_local_embeds * hard_attention_value1.unsqueeze(-1)).sum(-2)  # bs  1024
        head = self.gobal_attention(question_embeds, head1.unsqueeze(1), head1.unsqueeze(1))
        return head
    
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

class BiVision_VQA11(nn.Module):
    def __init__(self, 
                 image_size = 480,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:0"
                 ):
        super().__init__()
        self.device = device
        self.encoder_config = BertConfig.from_json_file(med_config)
        # self.vae1 = VAE(input_dim=self.encoder_config.hidden_size)
        # self.vae2 = VAE(input_dim=self.encoder_config.hidden_size)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)
        # self.resnet = Transfer(self.encoder_config)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        # self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 
        self.text_encoder = BertModel(config=modelConfig) 
        # pretrained_dict = torch.load('./ckpt/pretrain/bert_39_best.pt')
        # model_dict = self.text_encoder.state_dict()
        # # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fusion_layer' not in k and "classifier.2" not in k)}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        # model_dict.update(pretrained_dict)
        # self.text_encoder.load_state_dict(model_dict, strict=False)
        
        # self.local_encoder = LXRTEncoder(modelConfig)
        self.transformers = Transformer(self.encoder_config)
        # self.gobal_encoder = LXRTEncoder(modelConfig)
        self.sentence_bert = SentenceTransformer('all-distilroberta-v1')
        
        self.encoder_config.encoder_width = vision_width
        # self.v_att_proj1 = nn.Linear(768, 2048)
        
        self.v_att_proj1 = nn.Linear(768, 1024)
        self.v_att_proj2 = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)
        
        # self.mfh_proj = nn.Linear(2048, 1024)

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
         #self.CoAtt = CoAtt(self.encoder_config)
        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.local_mfb = MFB(self.encoder_config, self.encoder_config.MFB_size, self.encoder_config.MFB_size)
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)
        
        # self.classifier = nn.Sequential(nn.Linear(1024, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        self.classifier1 = nn.Sequential(nn.Linear(768, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))

    def forward(self, image, tokens, segment_ids, input_mask, answer=None, id2ans=None, ans_tokens_list=None, question=None, train=False, not_pretrain=True, val=False):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768

        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        sentence_embeds = self.sentence_bert.encode(question, device=self.device)
        sentence_embeds = torch.tensor(sentence_embeds).to(self.device)
        question_embeds = self.local_attention(question_embeds, sentence_embeds.unsqueeze(1), sentence_embeds.unsqueeze(1))
        
        # question_embeds, image_local_embeds = self.local_encoder(question_embeds, input_mask, image_local_embeds)
        # question_embeds, image_global_embeds = self.local_encoder(question_embeds, input_mask, image_global_embeds)
        
        h, attn_score = self.transformers(image_local_embeds, question_embeds, input_mask)
        h = h.mean(0)   # [4,8,60,768] -> [8,60,768]
        # h = h[-1]
        image_local_embeds, question_embeds = h[:, 1:39,:], h[:, 39:,:]
        question_embeds = torch.cat((h[:, 0, :].unsqueeze(1), question_embeds), dim=1)
        
        # image_local_embeds = self.v_att_proj1(image_local_embeds)
        # # image_global_embeds = self.v_att_proj2(image_global_embeds)
        # question_embeds = self.l_att_proj(question_embeds)
        
        sim_matrix_v2l1 = torch.matmul(image_local_embeds, question_embeds.transpose(1,2))  # bs * 362 * 20
        kg_output1, k1 = torch.topk(sim_matrix_v2l1, dim=-1, k=1)           # bs * 362 * 1
        hard_attention_value1 = gumbel_softmax(kg_output1.squeeze())        # bs * 362
        head1 = (image_local_embeds * hard_attention_value1.unsqueeze(-1)).sum(-2)  # bs  1024
        
        head = self.gobal_attention(question_embeds, head1.unsqueeze(1), head1.unsqueeze(1))
        if not_pretrain:
            head = head.mean(1)
        #head = self.l_att_proj(h[:, 0, :])

        logits = self.classifier1(head)
        return logits

class BiVision_VQA12(nn.Module):
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
        # self.vae1 = VAE(input_dim=self.encoder_config.hidden_size)
        # self.vae2 = VAE(input_dim=self.encoder_config.hidden_size)

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)
        # self.resnet = Transfer(self.encoder_config)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        # self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 
        self.text_encoder = BertModel(config=modelConfig) 
        # self.sentence_bert = SentenceTransformer('all-distilroberta-v1')
        self.sentence_bert = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=self.device)
        # pretrained_dict = torch.load('./ckpt/pretrain/bert_39_best.pt')
        # model_dict = self.text_encoder.state_dict()
        # # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fusion_layer' not in k and "classifier.2" not in k)}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        # model_dict.update(pretrained_dict)
        # self.text_encoder.load_state_dict(model_dict, strict=False)
        
        # self.local_encoder = LXRTEncoder(modelConfig)
        self.transformers = Transformer(self.encoder_config)
        # self.gobal_encoder = LXRTEncoder(modelConfig)
        
        self.encoder_config.encoder_width = vision_width
        # self.v_att_proj1 = nn.Linear(768, 2048)
        
        self.v_att_proj1 = nn.Linear(768, 1024)
        self.v_att_proj2 = nn.Linear(768, 1024)
        self.l_att_proj = nn.Linear(768, 1024)
        
        # self.mfh_proj = nn.Linear(2048, 1024)

        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
         #self.CoAtt = CoAtt(self.encoder_config)
        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, \
            hidden_size=self.encoder_config.hidden_size, use_fc=True, w_size=self.encoder_config.seq_max_len)
        self.local_mfb = MFB(self.encoder_config, self.encoder_config.MFB_size, self.encoder_config.MFB_size)
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)
        
        # self.classifier = nn.Sequential(nn.Linear(1024, self.encoder_config.mlp_size),
        #                                 nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
        #                                 nn.Linear(self.encoder_config.mlp_size, output_size))
        self.classifier1 = nn.Sequential(nn.Linear(768, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        self.classifier2 = nn.Sequential(nn.Linear(768*2, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))
        
        self.loss_triplet= nn.TripletMarginLoss(margin=1.0, p=2)
        self.loss_ce = nn.CrossEntropyLoss()
        
        self.config = config
        ans_embeds_all = torch.load(config["answer_embeds_path"])
        self.ans_embeds_all = ans_embeds_all.to(device)

    def forward(self, image, tokens, segment_ids, input_mask, answer, id2ans, ans_tokens_list, question=None, train=False, not_pretrain=True, val=False):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768
        
        # image_global_embeds = self.resnet(image)  # batch_size*768

        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        sentence_embeds = self.sentence_bert.encode(question)
        sentence_embeds = torch.tensor(sentence_embeds, device=self.device)
        question_embeds = self.local_attention(question_embeds, sentence_embeds.unsqueeze(1), sentence_embeds.unsqueeze(1))
        
        # sentence_embeds = self.sentence_bert.encode(question)
        # question_embeds = torch.tensor(sentence_embeds).to(self.device)
        
        # question_embeds, image_local_embeds = self.local_encoder(question_embeds, input_mask, image_local_embeds)
        # question_embeds, image_global_embeds = self.local_encoder(question_embeds, input_mask, image_global_embeds)
        
        
        h, attn_score = self.transformers(image_local_embeds, question_embeds, input_mask)
        h = h.mean(0)   # [4,8,60,768] -> [8,60,768]
        # h = h[-1]
        # image_local_embeds, question_embeds = h[:, 1:39,:], h[:, 39:,:]
        # question_embeds = torch.cat((h[:, 0, :].unsqueeze(1), question_embeds), dim=1)
        
        # image_local_embeds = self.v_att_proj1(image_local_embeds)
        # # image_global_embeds = self.v_att_proj2(image_global_embeds)
        # question_embeds = self.l_att_proj(question_embeds)
        
        #################33
        # sim_matrix_v2l1 = torch.matmul(image_local_embeds, question_embeds.transpose(1,2))  # bs * 362 * 20
        # kg_output1, k1 = torch.topk(sim_matrix_v2l1, dim=-1, k=1)           # bs * 362 * 1
        # hard_attention_value1 = gumbel_softmax(kg_output1.squeeze())        # bs * 362
        # head1 = (image_local_embeds * hard_attention_value1.unsqueeze(-1)).sum(-2)  # bs  1024
        
        # head = self.gobal_attention(question_embeds, head1.unsqueeze(1), head1.unsqueeze(1))
        head = h
        head = head.mean(1)
        #head = self.l_att_proj(h[:, 0, :])

        logits = self.classifier1(head)
        loss_ce1 = self.loss_ce(logits, answer)
        
        # # ##########3
        tokens_list, segment_ids_list, input_mask_list, target, preds_top, logits3 = self.get_top_ans_embeds(logits, answer, id2ans, ans_tokens_list)
        tokens_list, segment_ids_list, input_mask_list, target = \
            tokens_list.to(self.device), segment_ids_list.to(self.device), input_mask_list.to(self.device), target.to(self.device)
        answer_embeds = self.text_encoder(input_ids=tokens_list, token_type_ids=segment_ids_list, position_ids=input_mask_list)[0]
            
        # answer_embeds = F.normalize(self.v_att_proj2(answer_embeds[:, 0, :]), dim=1)
        answer_embeds = F.normalize(answer_embeds[:, 0, :], dim=1)
        answer_embeds = answer_embeds.view(head.shape[0], -1, head.shape[1]) # [bs, 6, 768]
        target = target.view(answer_embeds.shape[0], -1)
        
        # with torch.no_grad():
        #     for i in range(len(answer_embeds)):
        #         self.ans_embeds_all[preds_top[i]] = answer_embeds[i, :-1, :]
        

        # head1 = head.repeat(1, answer_embeds.shape[2]).view(head.shape[0], -1, head.shape[1])   # # [bs*6, 1024] -> # [bs, 6, 1024]
        head1 = head.unsqueeze(1) # [bs, 768] -> [bs, 1, 768]
        head1 = F.normalize(head1, dim=2)
        
        similarity_matrix = torch.matmul(head1, answer_embeds.transpose(1, 2)).squeeze(1)  # [bs, 1, 768] * [bs, 768, 6] -> [bs, 1, 6] -> [bs, 6]
        # select and combine multiple positives
        # positives = similarity_matrix[target.bool()].view(target.shape[0], -1)
        contrastive_loss = self.contrastive_loss(head1, answer_embeds, similarity_matrix, target).to(self.device)

        # select only the negatives the negatives
        # negatives = similarity_matrix[~target.bool()].view(similarity_matrix.shape[0], -1)
        # negatives = self.info_nce_loss(similarity_matrix, target)
        negatives = get_negative_pair(answer_embeds, similarity_matrix, target)
        loss_triplet = self.loss_triplet(head1, answer_embeds[:, -1, :], negatives)
        
        head2 = head.repeat(1, answer_embeds.shape[1]).view(head.shape[0], -1, answer_embeds.shape[2])
        head3 = torch.cat((head2, answer_embeds), dim=2)
        logits2 = self.classifier2(head3)
        
        ans_size = logits2.shape[-1]
        loss_ce2 =  self.loss_ce(logits2.view(-1, ans_size), target.view(-1))    # [bs, 6, 2] -> [bs*6, 2]  [bs, 6] -> [bs*6]
        
        
        
        sim_matrix_ha = torch.matmul(head.unsqueeze(1), answer_embeds.transpose(1,2))  # bs * 362 * 20
        kg_output2, k2 = torch.topk(sim_matrix_ha, dim=-1, k=1)           # bs * 362 * 1
        hard_attention_value2 = gumbel_softmax(kg_output2.squeeze())        # bs * 362
        head2 = (head.unsqueeze(1) * hard_attention_value2.unsqueeze(-1)).sum(-2)  # bs  1024
        head2 = self.gobal_attention(answer_embeds, head2.unsqueeze(1), head2.unsqueeze(1))
        head2 = torch.cat((head2, head.unsqueeze(1)), dim=1).mean(1)
        
        logits = self.classifier1(head2)
        loss_ce3 = self.loss_ce(logits, answer)
        loss_ce = loss_ce1 + loss_ce2 + loss_ce3
        
        if val:
            # logits4 = torch.matmul(F.normalize(head, dim=1), self.ans_embeds_all.transpose(0, 1))
            # # loss_ce3 = self.loss_ce(logits4, answer)
            # logits = logits + logits4
            return logits
        # if val:
        #     logits2 = logits2[:, :-1, -1]
        #     a = torch.zeros([logits3.shape[0], logits3.shape[1]], dtype=torch.float).to(self.device)
        #     for i in range(len(preds_top)):
        #         a[i][preds_top[i]] = logits2[i]
                
        #     # logits3[preds_top] = logits2
        #     return a
        
        return logits, contrastive_loss, loss_triplet, loss_ce
    
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

def get_negative_pair(answer_embeds, similarity_matrix, target):
    dist = similarity_matrix
    dist[~target.bool()] = 0

    dist_max = dist.max(1)  # 去最大值
    negative = answer_embeds[:, dist_max.indices, :]
    return negative 

class bert_base(nn.Module):
    def __init__(self, 
                 med_config = 'configs/med_config.json',
                 output_size = 1024,
                 ):
        super().__init__()
        self.encoder_config = BertConfig.from_json_file(med_config)
        modelConfig = BertConfig.from_pretrained('./bert-base-uncased/bert-base-uncased-config.json')
        self.text_encoder = BertModel.from_pretrained('./bert-base-uncased/bert-base-uncased-pytorch_model.bin', config=modelConfig) 
        self.classifier = nn.Sequential(nn.Linear(768, self.encoder_config.mlp_size),
                                        nn.LayerNorm(self.encoder_config.mlp_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(self.encoder_config.mlp_size, output_size))

    def forward(self, tokens, segment_ids, input_mask):
        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]
        question_embeds = question_embeds.mean(1)
        logits = self.classifier(question_embeds)
        # logits = F.normalize(logits, dim=-1)
        return logits

class MARC(nn.Module):
    def __init__(self, 
                 model = None,
                 K = 1024,
                 device = "cuda:0"
                 ):
        super().__init__()

        self.model = model
        self.omega = torch.nn.Parameter(torch.ones(1, 30524))
        self.beta = torch.nn.Parameter(torch.zeros(1, 30524))

    def forward(self, image, tokens, segment_ids, input_mask, caption="", train=False, not_pretrain=False):
        with torch.no_grad():
            w_norm = torch.norm(self.model.classifier[-1].weight, dim=1)
            logits = self.model(image, tokens, segment_ids, input_mask, not_pretrain=True)
        logits = self.omega * logits + self.beta*w_norm
        return logits 

class mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, hid_emb):
        hid_emb = self.fc1(hid_emb)
        x = self.drop(hid_emb)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.drop(x)
        # x = nn.functional.normalize(x)
        return x, hid_emb
    
# class mlp(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, hidden_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, hid_emb):
#         hid_emb = self.fc1(hid_emb)
#         x = self.act(hid_emb)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x, hid_emb

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

class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()

        self.args = args
        self.model = torch_models.resnet152(pretrained=True)
        # for p in self.parameters():
        #     p.requires_grad=False
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1,1))
        self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1,1))
        self.conv5 = nn.Conv2d(256, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1,1))
        self.conv7 = nn.Conv2d(64, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, img):
        modules2 = list(self.model.children())[:-2]
        fix2 = nn.Sequential(*modules2)
        v_2 = self.gap2(self.relu(self.conv2(fix2(img)))).view(-1,self.args.hidden_size)    # [bs, 768]
        modules3 = list(self.model.children())[:-3]
        fix3 = nn.Sequential(*modules3)
        v_3 = self.gap3(self.relu(self.conv3(fix3(img)))).view(-1,self.args.hidden_size)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        v_4 = self.gap4(self.relu(self.conv4(fix4(img)))).view(-1,self.args.hidden_size)
        modules5 = list(self.model.children())[:-5]
        fix5 = nn.Sequential(*modules5)
        v_5 = self.gap5(self.relu(self.conv5(fix5(img)))).view(-1,self.args.hidden_size)
        modules7 = list(self.model.children())[:-7]
        fix7 = nn.Sequential(*modules7)
        v_7 = self.gap7(self.relu(self.conv7(fix7(img)))).view(-1,self.args.hidden_size)
        h = torch.cat((v_2.unsqueeze(1), v_3.unsqueeze(1), v_4.unsqueeze(1), v_5.unsqueeze(1), v_7.unsqueeze(1)), dim=1)
        # bs*5*768
        return h 
    
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()
        # base_model = BertModel.from_pretrained('bert-base-uncased')
        # bert_model = nn.Sequential(*list(base_model.children())[0:])
        # self.bert_embedding = bert_model[0]
        # self.embed = Embeddings(args)
        # self.trans = Transfer(args)
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers
        
    def forward(self, image_embeds, h, mask):
        h[:,1:38,:] = image_embeds
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
    
class BertLayer(nn.Module):
    def __init__(self,args, share='all', norm='pre'):
        super(BertLayer, self).__init__()
        self.share = share
        self.norm_pos = norm
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.drop1 = nn.Dropout(args.hidden_dropout_prob)
        self.drop2 = nn.Dropout(args.hidden_dropout_prob)
        if self.share == 'ffn':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = PositionWiseFeedForward(args)
        elif self.share == 'att':
            self.attention = MultiHeadedSelfAttention(args)
            self.proj = nn.Linear(args.hidden_size, args.hidden_size)
            self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
        elif self.share == 'all':
            self.attention = MultiHeadedSelfAttention(args)
            self.proj = nn.Linear(args.hidden_size, args.hidden_size)
            self.feedforward = PositionWiseFeedForward(args)
        elif self.share == 'none':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
            
    def forward(self, hidden_states, attention_mask, layer_num):
        if self.norm_pos == 'pre':
            if isinstance(self.attention, nn.ModuleList):
                attn_output, attn_scores = self.attention[layer_num](self.norm1(hidden_states), attention_mask)
                h = self.proj[layer_num](attn_output)
            else:
                h = self.proj(self.attention(self.norm1(hidden_states), attention_mask))
            out = hidden_states + self.drop1(h)
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](self.norm1(out))
            else:
                h = self.feedforward(self.norm1(out))
            out = out + self.drop2(h)
        if self.norm_pos == 'post':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](hidden_states, attention_mask))
            else:
                h = self.proj(self.attention(hidden_states, attention_mask))
            out = self.norm1(hidden_states + self.drop1(h))
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](out)
            else:
                h = self.feedforward(out)
            out = self.norm2(out + self.drop2(h))
        return out, attn_scores

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadedSelfAttention,self).__init__()
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.drop = nn.Dropout(args.hidden_dropout_prob)
        self.scores = None
        self.n_heads = args.heads
        
    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
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

class PositionWiseFeedForward(nn.Module):
    def __init__(self,args):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size*4)
        self.fc2 = nn.Linear(args.hidden_size*4, args.hidden_size)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

