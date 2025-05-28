import torch
from torch import nn
from dit_blk import DiTBlock
from temporal_embedding import TimeEmbedding
from config import T


class DiT(nn.Module):
    def __init__(self, imgSize,
                        patchSize,
                        channels,
                        embSize,
                        labelNum,
                        ditNum,
                        headNum):
        super().__init__()
        self.patchSize = patchSize
        self.patchNum = imgSize // patchSize
        self.channel = channels

        # patchify
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels * patchSize ** 2,
                              kernel_size=patchSize,
                              padding=0,
                              stride=patchSize)
        self.patchEmb = nn.Linear(in_features=channels * patchSize ** 2,
                                  out_features=embSize)
        self.patchPosEmb = nn.Parameter(torch.rand(1, self.patchNum ** 2, embSize))

        # time embedding
        self.timeEmb = nn.Sequential(
            TimeEmbedding(embSize),
            nn.Linear(in_features=embSize, out_features=embSize),
            nn.GELU(),
            nn.Linear(in_features=embSize, out_features=embSize),
        )

        # label embedding
        self.labelEmb = nn.Embedding(num_embeddings=labelNum, embedding_dim=embSize)

        # DiT blk
        self.dits = nn.ModuleList([
            DiTBlock(embSize=embSize,
                     headNum=headNum)
            for _ in range(ditNum)
        ])

        # layer norm
        self.layerNorm = nn.LayerNorm(embSize)

        # linear back to patch
        self.l2p = nn.Linear(in_features=embSize,
                             out_features=channels * patchSize ** 2)


    def forward(self, x, t, y):
        # label embbeding
        y_emb = self.labelEmb(y)
        # time embedding
        t_emb = self.timeEmb(t)
        # condition embedding
        cond = y_emb + t_emb

        # patch embbeding
        x = self.conv(x)           # (batch, channels, patchNum, patchNum)
        x = x.permute(0, 2, 3, 1)  # (batch, patchNum, patchNum, channels)
        x = x.view(x.size(0), self.patchNum ** 2, x.size(3)) # (batch, patchNum ** 2, channels)

        x = self.patchEmb(x)  # (batch, patchNum ** 2, embSize)
        x = x + self.patchPosEmb  # (batch, patchNum ** 2, embSize)

        # DiT blks
        for dit in self.dits:
            x = dit(x, cond)

        # layer norm
        x = self.layerNorm(x)  # (batch, patchNum ** 2, embSize)

        # linear back to patch
        x = self.l2p(x)  # (batch, patchNum ** 2, channels * patchSize ** 2)

        # reshape to image
        x = x.view(x.size(0), self.patchNum, self.patchNum, self.channel, self.patchSize, self.patchSize)
        x = x.permute(0, 3, 1, 2, 4, 5)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(x.size(0), self.channel, self.patchNum * self.patchSize, self.patchNum * self.patchSize)
        return x


if __name__ == '__main__':
    model = DiT(imgSize=28,
                patchSize=4,
                channels=1,
                embSize=64,
                labelNum=10,
                ditNum=3,
                headNum=4)
    x = torch.rand(5, 1, 28, 28)
    t = torch.randint(0, T, (5,))
    y = torch.randint(0, 10, (5,))
    outputs = model(x, t, y)
    print(outputs.shape)
        
        