import math

import torch
from torch import nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, p_dropout, max_len=5000, base_val=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p_dropout)

        PE = torch.zeros((max_len, d_model))

        pos = torch.arange(max_len).unsqueeze(1)  # shape=(max_len, 1)
        i = torch.arange(max_len)  # shape=(d_model)
        div_term = torch.scalar_tensor(base_val).pow(2 * i / d_model)  # shape=(d_model)

        th = torch.div(pos, div_term)  # shape=(max_len, d_model)

        PE[:, 0::2] = torch.sin(th)
        PE[:, 1::2] = torch.cos(th)

        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)

    def forward(self, x):
        """
        :param x: torch.Tensor - shape=(batch, ls, d_embed)
        :return:
        """
        _, ls, _ = x.shape
        pe = Variable(self.PE[:, :ls], requires_grad=False)
        return pe


class Embedding(nn.Module):
    def __init__(self, d_model, vocab, p_dropout):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.positional_encoder = PositionalEncoding(d_model, p_dropout)

    def forward(self, x):
        x = self.lut(x) * math.sqrt(self.d_model) + self.positional_encoder(x)
        return x
