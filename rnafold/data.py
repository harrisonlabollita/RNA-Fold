import numpy as np
import RNA
import torch

#       Key
# --------------
#     ( ---> [1, 0, 0]
#     ) ---> [0, 1, 0]
#     . ---> [0, 0, 1]
# ----------------

def strAppend(s):
    output = ''
    output += s
    return output

def dot2num(array):
    new_array = []
    for elem in array:
        if elem == '(':
            new_array.append([1, 0, 0])
        elif elem == ')':
            new_array.append([0, 1, 0]
        else:
            new_array.append([0, 0, 1])
            
    new_array = np.array.reshape(len(array), 3)
    return new_array

def num2dot(array):
    dot = ''
    for elem in array:
        if elem == 0:
            dot += strAppend('(')
        elif elem == 1:
            dot += strAppend(')')
        else:
            dot += strAppend('.')
    return dot

def getRNA(filename):
    # input: file containing RNA sequences
    # output: array of RNA sequences
    sequences = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            sequences.append(line)
    return np.array(sequences)

def getDotBracket(filename):
    # input: file containing dot bracket representation
    # output: array of dot brackets
    dotbrackets = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            dotbrackets.append(line)
    return np.array(dotbrackets)


def RNAfoldPredict(sequences):
    # input: array of RNA sequences
    # output: array of predictions from RNAfold
    predictions = []
    for seq in sequences:
        pred, ener = RNA.fold(seq)
        predictions.append(pred)
    return np.array(predictions)

def getTrainingSets(file_seq, file_dots, batch_size):
    # input: filename1, filename2
    # output: torch datasets
    sequences = getRNA(file_seq)
    predictions = RNAfoldPredict(sequences)
    dotbrackets = getDotBracket(file_dots)
    # convert the dot bracket representation to number representation for training

    temp_pred = []
    for i in range(len(predictions)):
        temp_pred.append(dot2num(predictions[i]))

    temp_dot = []
    for i in range(len(dotbrackets)):
        temp_dot.append(dot2num(dotbrackets[i]))

    predictions = np.array(temp_pred)

    dotbrackets = np.array(temp_dot)

    predictions = torch.from_numpy(predictions).float()
    dotbrackets = torch.from_numpy(dotbrackets).float()

    N = int(0.8*len(predictions))

    training_set = torch.utils.data.TensorDataset(predictions[0:N], dotbrackets[0:N])
    testing_set = torch.utils.data.TensorDataset(predictions[N+1:len(predictions)], dotbrackets[N+1:len(dotbrackets)])

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
