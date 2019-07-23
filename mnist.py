import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

#setup: initialize variables and download/transform dataset
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

globalMean = 0.1307
stanDev = 0.3081

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (globalMean,), (stanDev,)) 
                             ])),
    batch_size=batch_size_train, shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (globalMean,), (stanDev,))
                             ])),
    batch_size=batch_size_test, shuffle=True, pin_memory=True)

#define network. 2 conv-2 FC/linear-2 relu-dropout-relu-dropout-softmax loss
class Net(nn.Module): #all trainable layers
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):#functions, no training
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

#network and optimizer initialize
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

#arrays to view progress/accuracy
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

#train code
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad() #pytorch accumulates gradients
    output = network(data) #forward pass
    loss = F.nll_loss(output, target) #loss
    loss.backward() #backward pass
    optimizer.step() #adjust weights

	#keeps track of progress/accuracy
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
     # (not working) torch.save(network.state_dict(), '/results/model.pth') #saves internal state 
     # (not working) torch.save(optimizer.state_dict(), '/results/optimizer.pth') #saves internal state 

#test code
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad(): #avoids storing computations done
    for data, target in test_loader:
      output = network(data) #forward pass/output
      test_loss += F.nll_loss(output, target, size_average=False).item() #sums up all test loss
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset) #average test loss percent
  test_losses.append(test_loss)
  #prints accuracy
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

#run test before train
test()
#train network - check with test set
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()


