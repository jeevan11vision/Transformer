import torch.nn as nn
import copy

from attention import MultiHeadedAttention
from decoder import Decoder, DecoderLayer
from embeddings import Embeddings, PositionalEncoding
from encoder import Encoder, EncoderLayer
from positionwise_feedforward import PositionwiseFeedForward


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        result = self.decode(memory, src_mask, tgt, tgt_mask)
        return result

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, p_dropout=0.1):
    c = copy.deepcopy

    position_encoder = PositionalEncoding(d_model, p_dropout)

    model = EncoderDecoder(
        encoder=Encoder(d_model, d_ff, p_dropout),
        decoder=Decoder(d_model, d_ff, h, p_dropout),
        src_embed=nn.Sequential([Embeddings(d_model, src_vocab), c(position_encoder)]),
        tgt_embed=nn.Sequential([Embeddings(d_model, tgt_vocab), c(position_encoder)]),
        generator=nn.Sequential([nn.Linear(d_model, tgt_vocab), nn.Softmax(dim=-1)]),
    )

    return model
