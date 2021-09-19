from torch import nn as nn

from attention import MultiHeadedAttention
from positionwise_feedforward import PositionwiseFeedForward
from utils import clones
from layer_noramalization import LayerNorm


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, p_dropout, N):
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(d_model, d_ff, p_dropout), N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, p_dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, p_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, p_dropout)
        self.dropouts = [nn.Dropout(p_dropout) for _ in range(2)]
        self.layer_norms = [LayerNorm(d_model) for _ in range(2)]

    def forward(self, x, mask):
        x = x + self.dropouts[0](self.self_attn(x, x, x, mask))
        x = self.layer_norms[0](x)
        x = x + self.dropouts[1](self.feed_forward(x))
        x = self.layer_norms[1](x)
        return x