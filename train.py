import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from model import Generator, Critic
from utils import gumbel_softmax_sample
from config import *


def gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1).to(DEVICE)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train(onehot_data_path=None, save_dir='./epoch'):
    if onehot_data_path is None:
        onehot_data_path = INPUT_PATH_ONEHOT

    real_data = torch.tensor(np.load(onehot_data_path), dtype=torch.float32)
    loader = DataLoader(TensorDataset(real_data), batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Critic().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))

    for epoch in range(EPOCHS):
        for real, in loader:
            real = real.to(DEVICE)
            curr_batch_size = real.size(0)

            for _ in range(CRITIC_ITER):
                z = torch.randn(curr_batch_size, NOISE_DIM).to(DEVICE)
                fake = gumbel_softmax_sample(G(z)).detach()

                d_loss = -torch.mean(D(real)) + torch.mean(D(fake))
                gp = gradient_penalty(D, real, fake)
                d_loss += LAMBDA_GP * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            z = torch.randn(curr_batch_size, NOISE_DIM).to(DEVICE)
            fake = gumbel_softmax_sample(G(z))
            g_loss = -torch.mean(D(fake))

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        if (epoch + 1) % 10 == 0:
            import os
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'{epoch}generator.pth')
            torch.save(G.state_dict(), path)

    G.eval()
    return G