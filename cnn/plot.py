import numpy as np
import matplotlib.pyplot as plt

def plotmodel_loss(epochs, loss, val_loss, title):
    epo = np.linspace(1, epochs, epochs)
    plt.figure()
    plt.title(title)
    plt.grid(True, linestyle = ':', linewidth = 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epo, loss, linewidth = 1, label = 'Training Loss')
    plt.plot(epo, val_loss, linewidth = 1, label = 'Val Loss')
    plt.legend(loc='best')
    plt.show()

def plotmodel_acc(epochs, acc, val_acc, title):
    epo = np.linspace(1, epochs,epochs)
    plt.figure()
    plt.title(title)
    plt.grid(True, linestyle = ':', linewidth = 1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epo, acc, linewidth = 1, label = 'Training Accuracy')
    plt.plot(epo, val_acc, linewidth = 1, label = 'Val Accuracy')
    plt.legend(loc='best')
    plt.show()
