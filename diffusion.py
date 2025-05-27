import torch
from config import T
from dataset import MNIST
from matplotlib import pyplot as plt


betas = torch.linspace(1e-4, 0.02, T)
aplhas = 1 - betas
alphas_cumprod = torch.cumprod(aplhas, dim=-1)
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), dim=-1)
variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # denoise

# forward add noise
def forward_add_noise(x, t):
    noise = torch.rand_like(x)  # add noise to each image at time t
    batch_alpha_cumprod = alphas_cumprod[t].view(x.size(0), 1, 1, 1)
    # based on formula, generate noised image at timestep tdirectly
    x = torch.sqrt(batch_alpha_cumprod) * x + torch.sqrt(1 - batch_alpha_cumprod) * noise
    return x, noise


if __name__ == '__main__':
    dataset = MNIST()

    x = torch.stack((dataset[0][0], dataset[1][0]), dim=0)

    # original image
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(x[0].permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(x[1].permute(1, 2, 0))
    plt.show()

    # randomly sample timestep
    t = torch.randint(0, T, size=(x.size(0),))
    print(f'timestep: {t}')

    # add noise
    x = x * 2 - 1
    x, noise = forward_add_noise(x, t)
    print('x:', x.size())
    print('noise:', noise.size())

    # added noise image
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(((x[0] + 1)/2).permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(((x[1] + 1)/2).permute(1, 2, 0))
    plt.show()