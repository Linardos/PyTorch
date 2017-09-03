import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from skimage import io, transform
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''
STEP 1: LOADING DATASET
'''

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

img = train_dataset[0][0].numpy().squeeze()
plt.imshow(img)
plt.show()

TD = Variable(train_dataset[0][0])
Convo = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)

Dconv = nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)


batch_size = 100
N = 1
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(256* 4 * 4, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.cnn1(x))
        out = self.maxpool(out)
        out = self.relu(self.cnn2(out))
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #Introduce the input
        self.bn0 = nn.BatchNorm1d(512*4*4)
        self.fc1 = nn.Linear(1, 512*4*4)
        self.dc1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.dc2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.dc3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.dc4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(32)
        self.dc5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.bn0(self.fc1(Z))
        out = out.view(out.size(0), 512, 4, 4)
        out = self.relu(self.bn1(self.dc1(out)))
        out = self.relu(self.bn2(self.dc2(out)))
        out = self.relu(self.bn3(self.dc3(out)))
        out = self.relu(self.bn4(self.dc4(out)))
        out = self.dc5(out)
        return out

Z=Variable(torch.rand(100)).unsqueeze(0).t()
G = Generator()
generated = G(Z)
img =generated[0][0].data.numpy().squeeze()
plt.imshow(img)
plt.show()
