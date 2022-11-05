import torch
import numpy as np
import rna2matrix as matrix
import model as rnaConvNet
import load_data as load
import nussinov as nussinov
import RNA
import matplotlib.pyplot as plt

def getFile(filename):
    array = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.rstrip()
            array.append(line)
    return np.array(array)

def out2dot(matrix):
    out = ''
    for i in range(len(matrix)):
        val, index = torch.max(matrix[i], 1)
        for i in range(len(index)):
            if index[i] == 0:
                out += '('
            elif index[i] == 1:
                out += ')'
            else:
                out += '.'
    return out

def testModel(seq, tgt):
    bad = 0
    outputs = []
    accuracies = []
    if len(seq) != len(tgt):
        print('Length of files do not match!')

    for i in range(len(seq)):
        test = matrix.RNAmatrix(seq[i])
        test = torch.from_numpy(test)
        test = test.view(1, 1, 30, 30)
        output = model(test.type('torch.FloatTensor'))
        output = out2dot(output)

        if badPrediction(output) == False:
            bad += 1

        outputs.append(output)
        accuracies.append(modelAccuracy(output, tgt[i]))

    return np.array(accuracies), np.array(outputs), bad

def modelAccuracy(pred, tgt):
    count = 0
    for i in range(len(pred)):
        if pred[i] == tgt[i]:
            count +=1
    count /= len(pred)
    return count



PATH = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/cnn code/cnn_trained_model.pt'

test_seq = getFile('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_src.txt')
test_tgt = getFile('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_tgt.txt')

model = rnaConvNet.rnaConvNet(30, 3)
model.load_state_dict(torch.load(PATH))
model.eval()

accuracies, outputs = testModel(test_seq, test_tgt)


print('Model Accuracy:', np.mean(accuracies))

plt.figure()
plt.hist(accuracies, bins ='auto', density= True, histtype = 'bar', rwidth = 0.9, color = 'black', alpha = 0.9, linewidth = 1)
plt.grid(True, linestyle = ':', linewidth = 1)
plt.title('CNN Model Histogram')
plt.xlabel('% of correct predictions/ sequence')
plt.ylabel('Frequency')
plt.text(0.4, 3.5, 'Testing on 1000 sequences')
plt.text(0.4, 3, 'Average: %0.2f' %(np.mean(accuracies)))
plt.show()
# This is a comment
