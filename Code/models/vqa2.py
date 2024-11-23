import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertTokenizer, BertModel

from models.vit import VisionTransformer, interpolate_pos_embed
from models.resnet import resnet152 
from models.med import BertConfig, BertLMHeadModel
from models.multi_attn import MultiHeadAttention

class BiVision_VQA2(nn.Module):
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

        self.vit_encoder, vision_width = create_vit(vit, image_size, self.encoder_config.patch_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.resnet_encoder = resnet152(num_classes=self.encoder_config.hidden_size)

        self.encoder_config.encoder_width = vision_width

        self.text_encoder = BertModel.from_pretrained('bert-base-uncased') 
        self.gru = nn.GRU(self.encoder_config.encoder_width, self.encoder_config.hidden_size, batch_first=True)

        self.gru_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=1)
        self.local_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)
        self.gobal_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)
        # self.semantics_attention = MultiHeadAttention(d_model=self.encoder_config.hidden_size, h=2, hidden_size=self.encoder_config.hidden_size, w_size=self.encoder_config.seq_max_len)

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.encoder_config.hidden_size*2, out_features=self.encoder_config.mlp_size_1))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.encoder_config.hidden_dropout_prob))
        # self.fusion.add_module('fusion_layer_1_activation', nn.ReLU())
        self.fusion.add_module('fusion_layer_2', nn.Linear(in_features=self.encoder_config.mlp_size_1, out_features=self.encoder_config.mlp_size_2))
        self.fusion.add_module('fusion_layer_2_dropout', nn.Dropout(self.encoder_config.hidden_dropout_prob))
        self.fusion.add_module('fusion_layer_2_activation', nn.ReLU())
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.encoder_config.mlp_size_2, out_features=output_size))
        # self.fusion.add_module('fusion_layer_norm', nn.LayerNorm(output_size))

    def forward(self, image, tokens, segment_ids, input_mask, caption="", train=True):
        image_local_embeds = self.vit_encoder(image)  # batch_size*16*768
        # image_global_embeds = self.resnet_encoder(image)  # batch_size*768
        # image_global_embeds = image_global_embeds.unsqueeze(1)

        # caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=self.encoder_config.seq_max_len, 
        #                           return_tensors="pt").to(image.device) 
        question_embeds = self.text_encoder(input_ids=tokens, token_type_ids=segment_ids, position_ids=input_mask)[0]

        question_embeds, _ = self.gru_forward(question_embeds, 
                                            image_local_embeds[:,0,:].unsqueeze(1), 
                                            self.gru.weight_ih_l0,
                                            self.gru.weight_hh_l0,
                                            self.gru.bias_ih_l0,
                                            self.gru.bias_hh_l0)
        
        local_output = self.local_attention(question_embeds, image_local_embeds[:,1:,:], image_local_embeds[:,1:,:])   # bs*seq_len*768
        gobal_output = self.gobal_attention(question_embeds, image_local_embeds[:,0,:].unsqueeze(1), image_local_embeds[:,0,:].unsqueeze(1))
        # gobal_output = self.gobal_attention(question_embeds, image_global_embeds[:][0][:], image_global_embeds[:][0][:]) # bs*seq_len*768
        # seman_output = self.semantics_attention(question, caption, caption)     # bs*seq_len*768
        
        # h = torch.cat(local_output.squeeze(1), gobal_output.squeeze(1), seman_output.squeeze(1), dim=1)
        h = torch.cat((local_output.squeeze(1), gobal_output.squeeze(1)), dim=1)
        output = self.fusion(h)
        # logits = F.softmax(output, dim=-1) # bs*ans_size
        return output 
    
    def gru_forward(self, input, image_embeds, w_ih, w_hh, b_ih, b_hh):
        bs,T,i_size=input.shape
        initial_states = torch.rand(bs, self.encoder_config.hidden_size).to(self.device)
        output = torch.zeros(bs, T, i_size).to(self.device)
        h_size=initial_states.shape[-1]
        h_0=initial_states

        batch_w_ih=w_ih.unsqueeze(0).tile(bs,1,1)
        batch_w_hh=w_hh.unsqueeze(0).tile(bs,1,1)

        prev_h=h_0

        for t in range(T):
            x=input[:,t,:]
            x = self.gru_attention(x.unsqueeze(1), image_embeds, image_embeds).squeeze()
            if bs == 1:
                x = x.unsqueeze(0)
            w_times_x = torch.bmm(batch_w_ih,x.unsqueeze(-1)).squeeze(-1)
            w_times_h = torch.bmm(batch_w_hh,prev_h.unsqueeze(-1)).squeeze(-1)
            r = torch.sigmoid(w_times_x[:,:h_size]+w_times_h[:,:h_size]+b_ih[:h_size]+b_hh[:h_size])
            z = torch.sigmoid(w_times_x[:,h_size:2*h_size]+w_times_h[:,h_size:2*h_size]+\
                b_ih[h_size:2*h_size]+b_hh[h_size:2*h_size])
            n = torch.tanh(w_times_x[:,2*h_size:3*h_size]+b_ih[2*h_size:3*h_size]+\
                r*(w_times_h[:,2*h_size:3*h_size]+b_hh[2*h_size:3*h_size]))
            prev_h = (1-z)*n+z*prev_h
            output[:,t,:] = prev_h
        return output, prev_h

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