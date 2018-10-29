import torch.nn as nn
import numpy as np
import torch


class Generator(nn.Module):
    def __init__(self, batchnorm, latent, n_classes, img_shape):
        super(Generator, self).__init__()

        if batchnorm:
            self.model = nn.Sequential(
                nn.Linear(latent + n_classes, 128),
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
                nn.Linear(latent + n_classes, 128),
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

    def forward(self, z, labels, img_shape):
        image_and_label = torch.cat((z, labels), dim=1)
        img = self.model(image_and_label)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, batchnorm, n_classes, img_shape):
        super(Discriminator, self).__init__()
        if batchnorm:
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape) + n_classes), 512),
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
                nn.Linear(int(np.prod(img_shape) + n_classes), 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        image_and_label = torch.cat((img_flat, labels), dim=1)
        prob = self.model(image_and_label)
        return prob


class Generator_Conv(nn.Module):
    def __init__(self, latent, n_classes, num_filters, channels):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent + n_classes, num_filters * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(num_filters * 8),
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(num_filters * 4),
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(num_filters * 2),
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(num_filters),
            nn.ConvTranspose2d(num_filters, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        image_and_label = torch.cat((z, labels), dim=1)
        img = self.model(image_and_label)
        return img


class Discriminator_Conv(nn.Module):
    def __init__(self, channels, num_filters, n_classes):
        super(Discriminator, self).__init__()

        self.model_image = nn.Sequential(
            nn.Conv2d(channels, num_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_filters * 2),
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_filters * 4),
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(num_filters * 8)
        )

        self.model_labels = nn.Sequential(
            nn.Linear(n_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(num_filters * 8 * 4 * 4 + 1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels, num_filters):
        x_image = self.model_image(img)
        x_image = x_image.view(-1, num_filters * 8 * 4 * 4)
        x_labels = self.model_labels(labels)
        # concatenate image and labels
        image_and_label = torch.cat((x_image, x_labels), dim=1)
        prob = self.final(image_and_label)
        return prob