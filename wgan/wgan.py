# import needed pytorch modules
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from net import Generator, Critic

# import other libraries
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='name of the dataset')
parser.add_argument('--dataroot', type=str, default='data-lsun', help='path for the data')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='learning rate for the optimizer')
parser.add_argument('--size', type=int, default=64, help='height for the image')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--latent', type=int, default=100, help='number of dimensions in latent space from which we generate the input noise')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads')
parser.add_argument('--sample_interval', type=int, default=391, help='interval betwen image samples')
parser.add_argument('--num_filters', type=int, default=64, help='number of filters')
parser.add_argument('--num_critic', type=int, default=5, help='every how many iterations you train the generator')
parser.add_argument('--clip_value', type=float, default=0.01, help='parameter for clipping the value')
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

# create the objects for two networks and for the two optimizers
generator = Generator(latent=opt.latent, channels=opt.channels, num_filters=opt.num_filters)
critic = Critic(channels=opt.channels, num_filters=opt.num_filters)
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.learning_rate)
optimizer_C = torch.optim.RMSprop(critic.parameters(), lr=opt.learning_rate)

# put the nets on gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator, critic = generator.to(device), critic.to(device)
generator.apply(weights_init)
critic.apply(weights_init)

if opt.dataset == 'cifar10':
    print(ROOT_DIR + "/cifar")
    if not os.path.isdir(ROOT_DIR + "/cifar"):
        os.mkdir(ROOT_DIR + "/cifar")
elif opt.dataset == 'LSUN':
    if not os.path.isdir(ROOT_DIR + "/bedrooms"):
        os.mkdir(ROOT_DIR + "/bedrooms")

# start training
current_epoch = 0
gen_iterations = 0
for epoch in range(opt.n_epochs):
    # train the critic
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        for p in critic.parameters():
            p.requires_grad = True

        if gen_iterations < 25 or gen_iterations % 500 == 0:
            critic_iterations = 100
        else:
            critic_iterations = opt.num_critic
        j = 0
        while j < critic_iterations and i < len(dataloader):
            j += 1
            # Clip weights of critic
            for p in critic.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
            data = data_iter.next()
            real_images, _ = data
            real_images = real_images.to(device)
            i += 1
            z = torch.FloatTensor(np.random.normal(0, 1, (real_images.shape[0], opt.latent, 1, 1))).to(device)
            generated_images = generator(z).detach()
            optimizer_C.zero_grad()
            c_loss = -(torch.mean(critic(real_images)) - torch.mean(critic(generated_images)))
            c_loss.backward()
            optimizer_C.step()

        # train the generator
        optimizer_G.zero_grad()
        for p in critic.parameters():
            p.requires_grad = False

        z = torch.FloatTensor(np.random.normal(0, 1, (real_images.shape[0], opt.latent, 1, 1))).to(device)
        generated_images = generator(z)
        g_loss = -torch.mean(critic(generated_images))
        g_loss.backward()
        optimizer_G.step()
        gen_iterations += 1

    # print losses and save the images
    batches_done = epoch * len(dataloader) + i
    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
            c_loss.item(), g_loss.item()))
    if opt.dataset == 'cifar10':
        save_image(generated_images.data[:25], 'cifar/%d.png' % current_epoch, nrow=5, normalize=True)
    elif opt.dataset == 'LSUN':
        save_image(generated_images.data[:25], 'bedrooms/%d.png' % current_epoch, nrow=5, normalize=True)
    current_epoch += 1