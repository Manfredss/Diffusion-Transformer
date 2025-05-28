import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from dit import DiT
import os
from dataset import MNIST
from config import T
from diffusion import forward_add_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.mps.is_available() else "cpu")
print('Using device:', device)

dataset = MNIST()
model = DiT(imgSize=28,
            patchSize=4,
            channels=1,
            embSize=64,
            labelNum=10,
            ditNum=3,
            headNum=4)
model.to(device)

# load model
try:
    model.load_state_dict(torch.load('model.pth'))
except:
    pass

# setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()
<<<<<<< HEAD
EPOCH = 500
=======
EPOCH = 1000
>>>>>>> c243c90d14375dfc56d8836243aa4e7526d50675
batch_size = 300
progress = tqdm(total=EPOCH)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model.train()

iter_counter = 0
for epoch in range(EPOCH):
    for img, label in dataloader:
        # convert image pixel range from [0, 1] to [-1, 1]
        x = img * 2 - 1
        t = torch.randint(0, T, (img.size(0),))
        y = label

        x, noise = forward_add_noise(x, t)
        pred_noise = model(x.to(device), t.to(device), y.to(device))
        loss = criterion(noise.to(device), pred_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_counter % 1000 == 0:
            print(f">> Epoch: {epoch}, Iteration {iter_counter}: loss = {loss}")
            torch.save(model.state_dict(), f".model.pth")
            os.replace(".model.pth", "model.pth")
        iter_counter += 1
    progress.update(1)