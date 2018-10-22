# import needed pytorch modules
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from net import Generator, Discriminator

# import other libraries
import os
import argparse
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='name of the dataset')
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
parser.add_argument('--loss', type=str, default='cross-entropy', help='type of loss, supports cross entropy (as in DCGAN paper) and mean square error as in lcgan')
opt = parser.parse_args()

if opt.dataset == 'LSUN':
    dataset = torchvision.datasets.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                                        transform=transforms.Compose([
                                            transforms.Resize(opt.imageSize),
                                            transforms.CenterCrop(opt.imageSize),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
else:
    os.makedirs('../../data/cifar10', exist_ok=True)
    # first we download and prepare the dataset
    dataset = torchvision.datasets.CIFAR10(root='../data', download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(opt.size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.n_cpu)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# create the objects for loss function, two networks and for the two optimizers
if opt.loss == 'cross-entropy':
    adversarial_loss = torch.nn.BCELoss()
else:
    adversarial_loss = torch.nn.MSELoss()

generator = Generator(latent=opt.latent, channels=opt.channels, num_filters=opt.num_filters)
discriminator = Discriminator(channels=opt.channels, num_filters=opt.num_filters)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate, betas=(opt.beta_1, opt.beta_2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(opt.beta_1, opt.beta_2))

# put the nets on gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator, discriminator = generator.to(device), discriminator.to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

if opt.dataset == 'cifar10':
    if not os.path.isdir(ROOT_DIR + "/cifar"):
        os.mkdir(ROOT_DIR + "/cifar")
elif opt.dataset == 'LSUN':
    if not os.path.isdir(ROOT_DIR + "/bedrooms"):
        os.mkdir(ROOT_DIR + "/bedrooms")

# start training
current_epoch = 0
for epoch in range(opt.n_epochs):
    for i, (inputs, _) in enumerate(dataloader):
        inputs = inputs.to(device)

        # create the labels for the fake and real images
        real = torch.ones(inputs.size(0), requires_grad=False)
        fake = torch.zeros(inputs.size(0), requires_grad=False)
        real, fake = real.to(device), fake.to(device)

        # train the generator
        optimizer_G.zero_grad()
        z = torch.FloatTensor(np.random.normal(0, 1, (inputs.shape[0], opt.latent, 1, 1))).to(device)
        generated_images = generator(z)

        # measure the generator loss and do backpropagation
        g_loss = adversarial_loss(discriminator(generated_images), real)
        g_loss.backward()
        optimizer_G.step()

        # train the discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(inputs), real)
        fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # print losses and save the images
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                             d_loss.item(), g_loss.item()))
            if opt.dataset == 'cifar10':
                save_image(generated_images.data[:25], 'cifar/%d.png' % current_epoch, nrow=5, normalize=True)
            elif opt.dataset == 'LSUN':
                save_image(generated_images.data[:25], 'bedrooms/%d.png' % current_epoch, nrow=5, normalize=True)
            current_epoch += 1