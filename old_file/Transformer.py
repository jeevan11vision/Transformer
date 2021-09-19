import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SubLayerConnection(nn.Module):
    def __init__(self, size, p_dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(self.norm(x)))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, p_dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SubLayerConnection(size, p_dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = LayerNorm

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, p_dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SubLayerConnection(size, p_dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, self.feed_forward)
        x = self.sublayers[2](x, lambda x: self.src_attn(x, m, m, src_mask))
        return x


def subsequent_mask(size):
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 1


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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, p_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fflayers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        """
        :param x: torch.Tensor - shape=(batch, ls, d_model)
        :return:
        """
        x = self.fflayers(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


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
        x = x + pe
        return x


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, p_dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, p_dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, p_dropout)

    position_encoder = PositionalEncoding(d_model, p_dropout)

    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), p_dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), p_dropout), N),
        src_embed=nn.Sequential([Embeddings(d_model, src_vocab), c(position_encoder)]),
        tgt_embed=nn.Sequential([Embeddings(d_model, tgt_vocab), c(position_encoder)]),
        generator=nn.Sequential([nn.Linear(d_model, tgt_vocab), nn.Softmax(dim=-1)]),
    )
