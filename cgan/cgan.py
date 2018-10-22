# import needed pytorch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms

# import other libraries
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--beta_1', type=float, default=0.5, help='beta_1 for adam optimizer')
parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 for adam optimizer')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate for the optimizer')
parser.add_argument('--height', type=int, default=28, help='height for the image')
parser.add_argument('--width', type=int, default=28, help='width for the image')
parser.add_argument('--latent', type=int, default=100, help='number of dimensions in latent space from which we generate the input noise')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads')
parser.add_argument('--sample_interval', type=int, default=469, help='interval betwen image samples')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
opt = parser.parse_args()

# first we download and prepare the dataset
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

img_shape = (opt.channels, opt.height, opt.width)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class Generator(nn.Module):
    def __init__(self, batchnorm):
        super(Generator, self).__init__()

        if batchnorm:
            self.model = nn.Sequential(
                nn.Linear(opt.latent + opt.n_classes, 128),
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
                nn.Linear(opt.latent + opt.n_classes, 128),
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

    def forward(self, z, labels):
        image_and_label = torch.cat((z, labels), dim=1)
        img = self.model(image_and_label)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, batchnorm):
        super(Discriminator, self).__init__()
        if batchnorm:
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape) + opt.n_classes), 512),
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
                nn.Linear(int(np.prod(img_shape) + opt.n_classes), 512),
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


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

# create the objects for loss function, two networks and for the two optimizers
adversarial_loss = torch.nn.BCELoss()
generator = Generator(batchnorm=True)
discriminator = Discriminator(batchnorm=True)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate, betas=(opt.beta_1, opt.beta_2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(opt.beta_1, opt.beta_2))

# put the nets on gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator, discriminator = generator.to(device), discriminator.to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

# start training
current_epoch = 0
for epoch in range(opt.n_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = one_hot_embedding(labels, opt.n_classes).to(device)

        # create the labels for the fake and real images
        real = torch.ones(inputs.size(0), requires_grad=False)
        fake = torch.zeros(inputs.size(0), requires_grad=False)
        real, fake = real.to(device), fake.to(device)

        # train the generator
        optimizer_G.zero_grad()
        z = torch.FloatTensor(np.random.normal(0, 1, (inputs.shape[0], opt.latent))).to(device)
        generated_images = generator(z, labels)

        # measure the generator loss and do backpropagation
        g_loss = adversarial_loss(discriminator(generated_images, labels), real)
        g_loss.backward()
        optimizer_G.step()

        # train the discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(inputs, labels), real)
        fake_loss = adversarial_loss(discriminator(generated_images.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # print losses and save the images
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            z = torch.FloatTensor(np.random.normal(0, 1, (50, opt.latent))).to(device)
            labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]
            labels = torch.LongTensor(labels)
            labels = one_hot_embedding(labels, opt.n_classes).to(device)
            generated_images = generator(z, labels)
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                             d_loss.item(), g_loss.item()))
            save_image(generated_images.data, 'images/%d.png' % current_epoch, nrow=5, normalize=True)
            current_epoch += 1



