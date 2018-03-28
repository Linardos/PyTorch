import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
dtype = torch.FloatTensor

#XOR toy data
x1 = np.array([1,1,0,0])
x2 = np.array([1,0,1,0])
y = np.array([0,1,1,0])
A = np.vstack((x1,x2))
threshold = 0.005

_input = Variable(torch.from_numpy(A.T).type(dtype), requires_grad = False)
target = Variable(torch.from_numpy(y).type(dtype), requires_grad = False)



class Net(nn.Module): #Net inherits all methods of the nn.Module

    def __init__(self, inc):
      #inherit the inits from the superclass nn.Module
        super(Net, self).__init__()
        # an affine operation: y = Wx + b, bias=True by default
        D = inc.size()[1]
        #fc stands for fully connected
        self.fc1 = nn.Linear(D, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net(_input)
params = list(net.parameters())
print(params[0])

for _ in range(500):
    #The output is 4 samples, 1 dimension.
    #The backward method requires a random tensor of such shape
    out = net.forward(_input)

    criterion = nn.MSELoss()
    loss = criterion(out, target)
    if _ == 499:
        #A peak on the operations
        print('Operations in backward order:\n')
        #creator lets us see how the object was created
        print(loss.creator)
        #we can go further back using "previous_functions"
        print(loss.creator.previous_functions[0][0])
        print(loss.creator.previous_functions[0][0].previous_functions[0][0])
        print(loss.creator.previous_functions[0][0].previous_functions[0][0].previous_functions[0][0])
        print('\n')
    loss.backward()

    learning_rate = 0.1
    #Update every parameter
    for f in net.parameters():
        f.data.sub_(f.grad.data*learning_rate)
    net.zero_grad()

print(out)
