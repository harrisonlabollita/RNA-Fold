# CONVOLUTIONAL NEURAL NETWORK IN PYTORCH
# Author: Harrison LaBollita
# Version: 3.0

import numpy as np
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F



class rnaConvNet(torch.nn.Module):

    def __init__(self, seq_length, num_classes):
        self.seq_length = seq_length
        self.num_classes = num_classes

        super(rnaConvNet, self).__init__()

        self.convLayer1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        self.convLayer2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        self.fullyConnect1 = torch.nn.Linear( 16*self.seq_length, self.seq_length)

        self.fullyConnect2 = torch.nn.Linear(self.seq_length, self.seq_length)

        self.fullyConnect3 = torch.nn.Linear(self.seq_length, self.num_classes)

        self.softmax = torch.nn.Softmax(dim = 2)




    def forward(self, x):
        # Output size = (batch_size, 16, seq_length +1 , seq_length + 1 )
        out = F.relu(self.convLayer1(x))
        out = self.pool1(out)

        # Output size = (batch_size, 16, seq_length, seq_length)
        out = F.relu(self.convLayer2(out))
        out = self.pool2(out)
        shape = out.size()
        # Output size = (batch_size, seq_length, seq_length*16)
        out = out.view(shape[0], self.seq_length, -1)

        # Output size = (batch_size, seq_length, seq_length)
        out = F.relu(self.fullyConnect1(out))

        # Output size = (batch_size, seq_length, seq_length)
        out = F.relu(self.fullyConnect2(out))

        # Output size = (batch_size, seq_length, num_classes)
        out = self.softmax(self.fullyConnect3(out))

        return out

def Loss():
    #loss = torch.nn.BCEWithLogitsLoss()
    loss = torch.nn.BCELoss()
    return loss

def Optimizer(net, learningRate, momentum):
    optimizer = optim.Adam(net.parameters(), learningRate, [momentum, momentum])
    return optimizer
