# import needed pytorch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms

# import other libraries
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar', help='name of the dataset')
parser.add_argument('--dataroot', type=str, default='data-lsun', help='path for the data')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--beta_1', type=float, default=0.5, help='beta_1 for adam optimizer')
parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 for adam optimizer')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate for the optimizer')
parser.add_argument('--size', type=int, default=64, help='height for the image')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--latent', type=int, default=100, help='number of dimensions in latent space from which we generate the input noise')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads')
parser.add_argument('--sample_interval', type=int, default=391, help='interval betwen image samples')
parser.add_argument('--num_filters', type=int, default=64, help='number of filters')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
opt = parser.parse_args()

if opt.dataset == 'LSUN':
    dataset = torchvision.datasets.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.n_cpu)
else:
    os.makedirs('../../data/cifar10', exist_ok=True)
    # first we download and prepare the dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.n_cpu)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(opt.latent + opt.n_classes, opt.num_filters * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(opt.num_filters * 8),
            nn.ConvTranspose2d(opt.num_filters * 8, opt.num_filters * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(opt.num_filters * 4),
            nn.ConvTranspose2d(opt.num_filters * 4, opt.num_filters * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(opt.num_filters * 2),
            nn.ConvTranspose2d(opt.num_filters * 2, opt.num_filters, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(opt.num_filters),
            nn.ConvTranspose2d(opt.num_filters, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        image_and_label = torch.cat((z, labels), dim=1)
        img = self.model(image_and_label)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model_image = nn.Sequential(
            nn.Conv2d(opt.channels, opt.num_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(opt.num_filters),
            nn.Conv2d(opt.num_filters, opt.num_filters * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(opt.num_filters * 2),
            nn.Conv2d(opt.num_filters * 2, opt.num_filters * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(opt.num_filters * 4),
            nn.Conv2d(opt.num_filters * 4, opt.num_filters * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(opt.num_filters * 8)
        )

        self.model_labels = nn.Sequential(
            nn.Linear(opt.n_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(opt.num_filters * 8 * 4 * 4 + 1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x_image = self.model_image(img)
        x_image = x_image.view(-1, opt.num_filters * 8 * 4 * 4)
        x_labels = self.model_labels(labels)
        # concatenate image and labels
        image_and_label = torch.cat((x_image, x_labels), dim=1)
        prob = self.final(image_and_label)
        return prob
        # return prob.view(-1, 1).squeeze(1)


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
generator = Generator()
discriminator = Discriminator()
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
        labels_unsqueeze = torch.unsqueeze(torch.unsqueeze(labels, 2), 2)

        # create the labels for the fake and real images
        real = torch.ones(inputs.size(0), requires_grad=False)
        fake = torch.zeros(inputs.size(0), requires_grad=False)
        real, fake = real.to(device), fake.to(device)

        # train the generator
        optimizer_G.zero_grad()
        z = torch.FloatTensor(np.random.normal(0, 1, (inputs.shape[0], opt.latent, 1, 1))).to(device)
        generated_images = generator(z, labels_unsqueeze)

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
            z = torch.FloatTensor(np.random.normal(0, 1, (50, opt.latent, 1, 1))).to(device)
            labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]
            labels = torch.LongTensor(labels)
            labels = one_hot_embedding(labels, opt.n_classes).to(device)
            labels_unsqueeze = torch.unsqueeze(torch.unsqueeze(labels, 2), 2)
            generated_images = generator(z, labels_unsqueeze)
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                             d_loss.item(), g_loss.item()))
            save_image(generated_images.data, 'images/%d.png' % current_epoch, nrow=5, normalize=True)
            current_epoch += 1