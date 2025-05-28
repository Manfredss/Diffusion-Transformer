from torch import nn
from config import T
import torch
import math


class TimeEmbedding(nn.Module):
    def __init__(self, embSize: int):
        super().__init__()
        self.halfDim = embSize // 2
        halfEmb = torch.exp(torch.arange(self.halfDim) * (-math.log(10000) / (self.halfDim - 1)))
        self.register_buffer('halfEmb', halfEmb)

    def forward(self, x):
        x = x.view(x.size(0), 1)
        halfEmb = self.halfEmb.unsqueeze(0).expand(x.size(0), self.halfDim)
        halfEmbX = halfEmb * x
        embX = torch.cat((halfEmbX.sin(), halfEmbX.cos()), dim=-1)
        return embX
    

if __name__ == '__main__':
    timeEmb = TimeEmbedding(16)
    x = torch.randint(0, T, (2,))
    embs = timeEmb(x)
    print(embs)


