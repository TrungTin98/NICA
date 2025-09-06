"""
This file contains NICA model.
"""

import torch
import torch.nn as nn
from utils import cosine


class NICA(nn.Module):
    def __init__(self, d_model, grow_steps=10, dropout=0.5):

        super(NICA, self).__init__()
        self.grow_steps = grow_steps

        self.linear = nn.Sequential(
            nn.Linear(2*d_model, 5*d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(5*d_model, d_model),
            nn.Dropout(dropout)
        )
        nn.init.zeros_(self.linear[0].bias)
        nn.init.zeros_(self.linear[3].bias)

    def forward(self, x):
        K = self.grow_steps

        for k in range(K):
            weighted_x = self.cosine_attention(x, x, x)
            state_x    = torch.cat((weighted_x, x), -1)
            dx         = self.linear(state_x)
            x          = x + dx

        return x

    def cosine_attention(self, queries, keys, values):
        q, k, v = queries.clone(), keys.clone(), values.clone()
        d = q.shape[-1]

        weights = cosine(q, k) * d**(0.5)
        attn_weights = nn.functional.softmax(weights, dim=-1)

        attn_output  = torch.matmul(attn_weights, v)
        return attn_output

