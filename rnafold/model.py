# Model for adjusting the output of RNAfold and fixing mistakes
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F


class foldNet(torch.nn.Module):

    def __init__(self, seq_length, hidden_size, batch_size):

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size

        super(foldNet, self).__init__()

        self.fullyConnect1 = torch.nn.Linear(self.batch_size, self.seq_length*3, self.hidden_size)

        self.fullyConnect2 = torch.nn.Linear(self.hidden_size, self.seq_length, self.batch_size)

        self.fullyConnect3 = torch.nn.Linear(self.batch_size, self.seq_length, 3)


    def forward(self, x):

        out = F.relu(self.fullyConnect1(x))
        out = F.relu(self.fullyConnect2(out))
        out = self.fullyConnect3(out)

        return out


def Loss():
    loss = torch.nn.BCEWithLogitsLoss()
    return loss

def Optimizer(net, learningRate):
    optimizer = optim.Adam(net.parameters(), learningRate)
    return optimizer

test = torch.rand(100, 30, 3)
test = test.view(100, -1)

net = foldNet(30, 150, 100 )

out = net(test)

print(out.size())
