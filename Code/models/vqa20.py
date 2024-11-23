import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertTokenizer, BertModel

import math
import numpy as np
import copy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import scipy.stats

import sys
sys.path.append('..')

from models.med import BertConfig
from swin_unet.networks.vision_transformer import SwinUnet as ViT_seg
from models.vit import VisionTransformer

class VQA10(nn.Module):
    def __init__(self, 
                 image_size = 224,
                 med_config = 'configs/med_config.json',
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 output_size = 1024,
                 device = "cuda:1",
                 config = None,
                 swin_cofig = None
                 ):
        super().__init__()
        self.device = device
        self.encoder_config = BertConfig.from_json_file(med_config)
        self.segmen = ViT_seg(swin_cofig, img_size=swin_cofig.img_size, num_classes=swin_cofig.seg_classes)
        self.vit_encoder1, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        
        
    def forward(self, img, image, tokens, segment_ids, input_mask, answer=None, id2ans=None, question=None): 
        
        img = self.segmen(img)
        img = torch.argmax(torch.softmax(img, dim=1), dim=1).squeeze(0)
        v_feat1 = self.vit_encoder1(image) 
        
        return 1, 2
        


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