import torch
from config import T
from dit import DiT
from matplotlib import pyplot as plt
from diffusion import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
<<<<<<< HEAD
# device = torch.device("mps" if torch.mps.is_available() else "cpu")
=======
device = torch.device("mps" if torch.mps.is_available() else "cpu")
>>>>>>> c243c90d14375dfc56d8836243aa4e7526d50675

def backword_denoise(model, x, y):
    steps = [x.clone(),]

<<<<<<< HEAD
    global alphas, alphas_cumprod, variance

    x = x.to(device)
    alphas = alphas.to(device)
=======
    global alpha, alphas_cumprod, variance

    x = x.to(device)
    aplha = alpha.to(device)
>>>>>>> c243c90d14375dfc56d8836243aa4e7526d50675
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
<<<<<<< HEAD
            mean = 1 / torch.sqrt(alphas[t].view(*shape)) * \
                    (x - (1 - alphas[t].view(*shape)) / torch.sqrt(1 - alphas_cumprod[t].view(*shape)) * noise) 
            if time != 0:
                x = mean + torch.randn_like(x) * torch.sqrt(variance[t].view(*shape))
=======
            mean = 1 / torch.sqrt(alpha[t].view(*shape)) * \
                        (x - (1 - alpha[t].view(*shape)) / torch.sqrt(1 - alphas_cumprod[t].view(*shape)) * noise) 
            if time != 0:
                x = mean + torch.rand_like(x) * torch.sqrt(variance[t].view(*shape))
>>>>>>> c243c90d14375dfc56d8836243aa4e7526d50675
            else:
                x = mean
            x = torch.clamp(x, -1.0, 1.0).detach()
            steps.append(x)
<<<<<<< HEAD
    return steps
=======
        return steps
>>>>>>> c243c90d14375dfc56d8836243aa4e7526d50675
    
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
<<<<<<< HEAD
y = torch.arange(start=0, end=10, dtype=torch.long)
=======
y = torch.arange(0, 10, dtype=torch.long)
>>>>>>> c243c90d14375dfc56d8836243aa4e7526d50675
# denoise
steps = backword_denoise(model, x, y)
# plot number
num_imgs = 20
# plot the progress of denoising
<<<<<<< HEAD
plt.figure(figsize=(15, 15))
=======
plt.figure(figsize=(10, 10))
>>>>>>> c243c90d14375dfc56d8836243aa4e7526d50675
for b in range(batch_size):
    for i in range(num_imgs):
        idx = int(T / num_imgs) * (i + 1)
        # convert back to [0, 1] and from tensor to PIL
<<<<<<< HEAD
        final_img = (steps[idx][b].to('cpu') + 1) / 2
        final_img = final_img.permute(1, 2, 0)
        plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
        plt.imshow(final_img)
plt.show()
=======
        final_img = ((steps[idx][b].to('cpu') + 1) / 2).permute(1, 2, 0)
        plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
        plt.imshow(final_img)
    plt.show()
>>>>>>> c243c90d14375dfc56d8836243aa4e7526d50675

