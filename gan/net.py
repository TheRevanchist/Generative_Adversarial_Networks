import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, batchnorm, latent, img_shape):
        super(Generator, self).__init__()

        if batchnorm:
            self.model = nn.Sequential(
                nn.Linear(latent, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(latent, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

    def forward(self, z, img_shape):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, batchnorm, img_shape):
        super(Discriminator, self).__init__()
        if batchnorm:
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        prob = self.model(img_flat)
        return prob