import torch
from torch.autograd import Variable
import numpy as np
import time
import model as model
import data as data
import plot as p

filename1 = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNAtrain_src.txt'
filename2 = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNAtrain_tgt.txt'

seq_length = 30
hidden_size = 100

batch_size = 100
epochs = 150
lr = 1e-3

def train(net, batch_size, epochs, lr):

    print('-'*30)
    print("  HYPERPARAMETERS  ")
    print('-'*30)
    print("Batch size = ", batch_size)
    print("Epochs = ", epochs)
    print("Learning rate = ", lr)
    print('-'*30)

    # function to call in data
    train_loader, test_loader = data.getTrainingSets(filename1, filename2, batch_size)

    # Create loss and optimizer functions
    loss = model.Loss()
    optimizer = model.Optimizer(net, lr)

    trainingStartTime = time.time()

    # Start training
    totalStep = len(train_loader)

    losses = []
    val_losses = []

    for epoch in range(epochs):

        runningLoss = 0.0
        totalTrainLoss = 0.0

        startTime = time.time()

        for i, (input, target) in enumerate(train_loader):

            input = Variable(input)
            target = Variable(target)

            outputs = net(input)
            loss_size = loss(outputs, target)
            losses.append(loss_size.item())

            optimizer.zero_grad()
            loss_size.backward()
            optimizer.step()

            runningLoss += loss_size.item()
            totalTrainLoss += loss_size.item()

        for i, (pred, real) in enumerate(test_loader):

            pred = Variable(pred)
            real = Variable(real)

            val_outputs = net(pred)
            val_loss_size = loss(val_outputs, real)
            val_losses.append(val_loss_size.item())

        print('Epoch: {}/{}, Loss: {:.4f}, Val loss: {:0.4f}, Time: {:0.2f}s'.format(epoch + 1, epochs,  loss_size.item(), val_loss_size.item(), time.time() - startTime))

    return losses, val_losses

network = model.foldNet(seq_length, hidden_size)

train_loss, validation_loss = train(network, batch_size, epochs, lr)

p.plotmodel(train_loss, validation_loss, 'Vienna RNA fold Model')
