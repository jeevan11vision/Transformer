from torch import nn as nn


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