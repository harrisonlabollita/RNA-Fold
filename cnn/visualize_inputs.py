import numpy as np
import matplotlib.pyplot as plt
import rna2matrix
import matplotlib
matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman, Times']})

source = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_src.txt'
target = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_tgt.txt'

def getRNA(filename ):
       sequences = []
       with open(filename) as f:
            for line in f:
                line = line.rstrip('\n')
                seq = []
                for char in line:
                    seq.append(str(char))
                sequences.append(seq)
       return sequences

sequences = getRNA(source)
rna2D = rna2matrix.seqs2matrices(sequences)




# Visualize

x_labels = sequences[1]
y_labels = sequences[1]
fig, ax = plt.subplots()
ax.imshow(rna2D[1])

ax.set_xticks(np.arange(rna2D[1].shape[1]) + 0.15, minor=False)
ax.set_yticks(np.arange(rna2D[1].shape[0]) + 0.15, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('both') # THIS IS THE ONLY CHANGE
ax.tick_params(axis='both', which='both', length=0)
ax.set_xticklabels(x_labels, minor=False)
ax.set_yticklabels(y_labels, minor=False)
plt.show()


for i in range(10):
    fig, ax = plt.subplots()
    x_labels = sequences[i]
    y_labels = sequences[i]
    plot = ax.imshow(rna2D[i])
    ax.set_xticks(np.arange(rna2D[i].shape[1]) + 0.15, minor=False)
    ax.set_yticks(np.arange(rna2D[i].shape[0]) + 0.15, minor=False)
    #ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_ticks_position('both') # THIS IS THE ONLY CHANGE
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels(x_labels, minor=False)
    ax.set_yticklabels(y_labels, minor=False)
    fig.colorbar(plot, ax = ax)
    plt.show()
