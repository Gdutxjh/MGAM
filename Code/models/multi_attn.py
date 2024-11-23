import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

def clones(module, N):
    """Product N identical layers."""
    # print("clones!")
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print("scores size: ", str(scores.size()))    # bs*multihead*seq_len*(768/multihead)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, hidden_size, w_size, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # create 4 linear layers
        self.attn = None
        self.w = nn.Parameter(torch.Tensor(1, w_size).normal_())
        self.b = nn.Parameter(torch.Tensor(1, hidden_size).normal_())
        self.dropout = nn.Dropout(p=dropout)
        self.w_size = w_size

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)  # bs*seq_len*768
        # print('Before transform query: ', str(query.size()))
        # (batch_size, seq_length, d_model)

        # 1) Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        query, key, value = [x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for x in (query, key, value)]
        # (batch_size, h, seq_length, d_k)
        # print('After transform query: ' + str(query.size()))  # bs*multihead*seq_len*768

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.linears[-1](x)

        if self.w_size > 1:
            x = torch.matmul(self.w, x)+self.b
        return x

# h = 8
# d_model = 512
# batch_size = 1024
# seq_length = 20
# patch_size = 16
# model = MultiHeadAttention(h, d_model)

# query = torch.randn([batch_size, seq_length, d_model])
# key = torch.randn([batch_size, patch_size, d_model])
# value = key

# print("Input size: ", str(query.size()))
# m = model(query, key, value)
# print("Output size: " + str(m.size()))