import torch
from config import T
from dit import DiT
from matplotlib import pyplot as plt
from diffusion import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.mps.is_available() else "cpu")

def backword_denoise(model, x, y):
    steps = [x.clone(),]

    global alphas, alphas_cumprod, variance

    x = x.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    variance = variance.to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        for time in range(T - 1, -1, -1):
            t = torch.full((x.size(0),), time).to(device)

            # predict noise at time t
            noise = model(x, t, y)
            # generate x at time t - 1
            shape = (x.size(0), 1, 1, 1)
            mean = 1 / torch.sqrt(alphas[t].view(*shape)) * \
                    (x - (1 - alphas[t].view(*shape)) / torch.sqrt(1 - alphas_cumprod[t].view(*shape)) * noise) 
            if time != 0:
                x = mean + torch.randn_like(x) * torch.sqrt(variance[t].view(*shape))
            else:
                x = mean
            x = torch.clamp(x, -1.0, 1.0).detach()
            steps.append(x)
    return steps
    
model = DiT(imgSize=28,
            patchSize=4,
            channels=1,
            embSize=64,
            labelNum=10,
            ditNum=3,
            headNum=4).to(device)
model.load_state_dict(torch.load("model.pth"))


batch_size = 10
# generate random noise
x = torch.randn(size=(batch_size, 1, 28, 28))
y = torch.arange(start=0, end=10, dtype=torch.long)
# denoise
steps = backword_denoise(model, x, y)
# plot number
num_imgs = 20
# plot the progress of denoising
plt.figure(figsize=(15, 15))
for b in range(batch_size):
    for i in range(num_imgs):
        idx = int(T / num_imgs) * (i + 1)
        # convert back to [0, 1] and from tensor to PIL
        final_img = (steps[idx][b].to('cpu') + 1) / 2
        final_img = final_img.permute(1, 2, 0)
        plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
        plt.imshow(final_img)
plt.show()

