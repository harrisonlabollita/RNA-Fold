import numpy as np
import matplotlib.pyplot as plt

def plotmodel(loss, val_loss, title):
    plt.figure()
    plt.title(title)
    plt.grid(True, linestyle = ':', linewidth = 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss, linewidth = 1, label = 'Training Loss')
    plt.plot(val_loss, linewidth = 1, label = 'Val Loss')
    plt.legend(loc='best')
    plt.show()
