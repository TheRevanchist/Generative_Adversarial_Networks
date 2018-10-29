# import needed pytorch modules
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from net import Generator_Conv as Generator
from net import Discriminator_Conv as Discriminator

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
generator = Generator(latent=opt.latent, n_classes=opt.n_classes, num_filters=opt.num_filters, channels=opt.channels)
discriminator = Discriminator(channels=opt.channels, num_filters=opt.num_filters, n_classes=opt.n_classes)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate, betas=(opt.beta_1, opt.beta_2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(opt.beta_1, opt.beta_2))

# put the nets on gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator, discriminator = generator.to(device), discriminator.to(device)
print(generator)
# generator.apply(weights_init)
# discriminator.apply(weights_init)

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
        g_loss = adversarial_loss(discriminator(generated_images, labels, opt.num_filters), real)
        g_loss.backward()
        optimizer_G.step()

        # train the discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(inputs, labels, opt.num_filters), real)
        fake_loss = adversarial_loss(discriminator(generated_images.detach(), labels, opt.num_filters), fake)
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
            save_image(generated_images.data, 'cifar/%d.png' % current_epoch, nrow=5, normalize=True)
            current_epoch += 1