import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from utils import clones


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, p_dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: torch.Tensor - shape=(batch, ls, d_model) - Query
        :param K: torch.Tensor - shape=(batch, ls, d_model) - Key
        :param V: torch.Tensor - shape=(batch, ls, d_model) - Value
        :param mask: torch.Tensor - shape=(1, ls, ls) - subsequent mask
        :return:
        """
        batch, ls, _ = Q.size

        Q, K, V = [
            layer(x).view(batch, ls, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (Q, K, V))
        ]
        x, self.attn = self.attention(Q, K, V, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch, ls, self.h * self.d_k)
        x = self.linears[-1](x)
        return x

    @staticmethod
    def attention(Q, K, V, mask=None, dropout=None):
        """

        :param Q: torch.Tensor - shape=(..., T, d_k) - Query
        :param K: torch.Tensor - shape=(..., T, d_k) - Key
        :param V: torch.Tensor - shape=(..., T, d_v) - Value
        :param mask: torch.Tensor - shape=(1, T, T) - subsequent mask
        :param dropout: Dropout layer
        :return:
        """
        d_k = Q.size[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # scores: shape=(..., T, T)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)  # p_attn: shape=(..., T, T)

        if dropout is not None:
            p_attn = dropout(p_attn)

        x = torch.matmul(p_attn, V)  # x : shape=(..., T, d_v)

        return x, p_attn