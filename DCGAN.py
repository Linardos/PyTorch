import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from skimage import io, transform
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

d_learning_rate = 2e-3
g_learning_rate = 2e-3
optim_betas = (0.9, 0.999)
d_steps = 5
g_steps = 1
num_epochs = 100
batch_size = 100
print_interval = 10

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



train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

"""
Batch Norm stabilizes the DCGAN. Do not use on : Discriminator 1st input and Generator output
Leaky Relu on Discriminator, Relu on Generator
Remember the output formula : (W = input, K=kernel_size, P=padding, S=stride)
O = (W-K+2P)/S for each convolution and
O = S(W-1)+K-2P for each transpose convolution (deconvolution)
based on these you can calculate the required dimensions on the Linear
"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256* 13 * 13, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.lrelu(self.bn1(self.cnn1(x)))
        out = self.lrelu(self.bn2(self.cnn2(out)))
        out = self.lrelu(self.bn3(self.cnn3(out)))
        out = self.lrelu(self.bn4(self.cnn4(out)))
        out = self.lrelu(self.bn5(self.cnn5(out)))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #Introduce the input
        self.bn0 = nn.BatchNorm1d(256*13*13)
        self.fc1 = nn.Linear(1, 256*13*13)
        self.dc1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.dc2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.dc3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)
        self.dc4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(16)
        self.dc5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn0(self.fc1(Z))
        out = out.view(out.size(0), 256, 13, 13)
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
print(generated.size())
D = Discriminator()
G = Generator()
criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)


for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        for i, (images, labels) in enumerate(train_loader):
            d_real_data = Variable(images)
            d_real_decision = D(d_real_data)
            d_real_error = criterion(d_real_decision, Variable(torch.ones(batch_size).unsqueeze(0).t()))  # ones = true
            d_real_error.backward()
            # compute/store gradients, but don't change params

            #  1B: Train D on fake
            Z=Variable(torch.rand(batch_size)).unsqueeze(0).t()
            d_fake_data = G(Z).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data)
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(batch_size).unsqueeze(0).t()))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            Z=Variable(torch.rand(batch_size)).unsqueeze(0).t()
            dg_fake_decision = D(G(Z))
            g_error = criterion(dg_fake_decision, Variable(torch.ones(batch_size).unsqueeze(0).t()))  # we want to fool, so pretend it's all genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters

    if epoch % print_interval == 0:
        generated = G(Z)
        img =generated[0][0].data.numpy().squeeze()
        plt.imshow(img)
        plt.show()
