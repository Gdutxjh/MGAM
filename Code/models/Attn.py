import torch
from torch import nn
import torch.nn.functional as F
import math

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) # [4, 12, 197, 20]
    # if mask is not None:
    #     # mask = mask[:, None, None, :].float()
    #     # mask = mask.unsqueeze(1)
    #     # scores = scores.masked_fill(mask == 0, -1e9)
    #     scores = torch.mul(scores, mask)
    
    scores = F.softmax(scores, dim=-1)  # [4, 12, 197, 197]
    if mask is not None:
        # scores = torch.mul(scores, mask.unsqueeze(-1))
        scores = torch.mul(scores, mask.unsqueeze(-2))
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)    # [4, 12, 197, 64]
    return output, scores


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
    
    def forward(self, q, k, v, mask=None, return_mask=False):
        
        bs = q.size(0)  # 4
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # [4, 20, 12, 64]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # [4, 197, 12, 64]
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) # [4, 20, 12, 64]
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)    # [4, 12, 20, 64]
        q = q.transpose(1,2)    # [4, 12, 197, 64]
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        output, scores = attention(q, k, v, self.d_k, mask, self.dropout)   # [4, 12, 20, 64]
        # concatenate heads and put through final linear layer
        concat = output.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        if return_mask:
            return output, scores
        return output
    

class MultiHead_Ques_Attention(nn.Module):
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
    
    def forward(self, q, k, v, mask=None, return_attn=False):
        
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # [4, 12, 1, 64]
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # [4, 12, 20, 64]
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) # [4, 12, 1, 64]
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        output, scores = attention(q, k, v, self.d_k, mask, self.dropout)   # [4, 12, 20, 64]
        # concatenate heads and put through final linear layer
        concat = output.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        if return_attn:
            return output, scores
        return output