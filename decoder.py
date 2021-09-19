from torch import nn as nn

from attention import MultiHeadedAttention
from positionwise_feedforward import PositionwiseFeedForward
from utils import clones
from layer_noramalization import LayerNorm


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, h, p_dropout, N):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(d_model, d_ff, h, p_dropout), N)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, p_dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, p_dropout)
        self.src_attn = MultiHeadedAttention(h, d_model, p_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, p_dropout)
        self.dropouts = [nn.Dropout(p_dropout) for _ in range(3)]
        self.layer_norms = [LayerNorm(d_model) for _ in range(3)]

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = x + self.dropouts[0](self.self_attn(x, x, x, tgt_mask))
        x = self.layer_norms[0](x)
        x = x + self.dropouts[1](self.self_attn(x, m, m, src_mask))
        x = self.layer_norms[1](x)
        x = x + self.dropouts[2](self.feed_forward(x))
        x = self.layer_norms[2](x)
        return x
