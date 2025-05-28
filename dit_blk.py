from torch import nn
import math
import torch


class DiTBlock(nn.Module):
    def __init__(self, embSize, headNum) -> None:
        super().__init__()
        self.embSize = embSize
        self.headNum = headNum

        # conditioning
        self.gamma1 = nn.Linear(embSize, embSize)
        self.beta1 = nn.Linear(embSize, embSize)
        self.alpha1 = nn.Linear(embSize, embSize)
        self.gamma2 = nn.Linear(embSize, embSize)
        self.beta2 = nn.Linear(embSize, embSize)
        self.alpha2 = nn.Linear(embSize, embSize)

        # layer norm
        self.layerNorm1 = nn.LayerNorm(embSize)
        self.layerNorm2 = nn.LayerNorm(embSize)

        # multi head attention
        self.wq = nn.Linear(embSize, headNum * embSize)
        self.wk = nn.Linear(embSize, headNum * embSize)
        self.wv = nn.Linear(embSize, headNum * embSize)
        self.lv = nn.Linear(headNum * embSize, embSize)

        # feed forward
        self.ff = nn.Sequential(nn.Linear(embSize, embSize * 4),
                                nn.ReLU(),
                                nn.Linear(embSize * 4, embSize))
        
    def forward(self, x, cond):
        # conditioning (batch, embSize)
        gamma1 = self.gamma1(cond)
        beta1 = self.beta1(cond)
        alpha1 = self.alpha1(cond)
        gamma2 = self.gamma2(cond)
        beta2 = self.beta2(cond)
        alpha2 = self.alpha2(cond)

        # layer norm
        y = self.layerNorm1(x)

        # scale and shift
        y = y * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)

        # attention
        q = self.wq(y)
        k = self.wk(y)
        v = self.wv(y)
        q = q.view(q.size(0), q.size(1), self.headNum, self.embSize).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.headNum, self.embSize).permute(0, 2, 3, 1)
        v = v.view(v.size(0), v.size(1), self.headNum, self.embSize).permute(0, 2, 1, 3)
        attn = q @ k / math.sqrt(q.size(2))
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(y.size(0), y.size(1), y.size(2) * y.size(3))
        y = self.lv(y)

        # scale
        y = y * alpha1.unsqueeze(1)

        # redisual
        y += x

        # layer norm
        y_ = self.layerNorm2(y)

        # scale and shift
        y_ = y_ * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)

        # pointwise feed forward
        y_ = self.ff(y_)

        # scale
        y_ = y_ * alpha2.unsqueeze(1)

        # redisual
        y_ += y

        return y_
    

if __name__ == '__main__':
    ditblk = DiTBlock(embSize=16, headNum=4)
    x = torch.rand((5, 49, 16))
    cond = torch.rand((5, 16))
    outputs = ditblk(x, cond)
    print(outputs.shape)
 