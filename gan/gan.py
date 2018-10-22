# import needed pytorch modules
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from net import Generator, Discriminator

# import other libraries
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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


# create the objects for loss function, two networks and for the two optimizers
batchnorm = True
adversarial_loss = torch.nn.BCELoss()
generator = Generator(batchnorm=batchnorm, latent=opt.latent, img_shape=img_shape)
discriminator = Discriminator(batchnorm=batchnorm, img_shape=img_shape)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate, betas=(opt.beta_1, opt.beta_2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(opt.beta_1, opt.beta_2))

# put the nets on device - if a cuda gpu is installed it will use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator, discriminator = generator.to(device), discriminator.to(device)

# initialize weights from random distribution with mean 0 and std 0.02
generator.apply(weights_init)
discriminator.apply(weights_init)

if batchnorm:
    if not os.path.isdir(ROOT_DIR + "/images-batchnorm"):
        os.mkdir(ROOT_DIR + "/images-batchnorm")
else:
    if not os.path.isdir(ROOT_DIR + "/images"):
        os.mkdir(ROOT_DIR + "/images")

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
        z = torch.FloatTensor(np.random.normal(0, 1, (inputs.shape[0], opt.latent))).to(device)
        generated_images = generator(z, img_shape)

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
            if batchnorm:
                save_image(generated_images.data[:25], 'images-batchnorm/%d.png' % current_epoch, nrow=5, normalize=True)
            else:
                save_image(generated_images.data[:25], 'images/%d.png' % current_epoch, nrow=5, normalize=True)
            current_epoch += 1