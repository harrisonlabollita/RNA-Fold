# load_data.py
# Input: sequences text file and dotbrackets text file, maximum sequence length
# Ouput: Numpy array of sequnces that are less than or equal to the max. Pads the sequences that are shorter than the maximum length


import numpy as np
import rna2matrix as matrix
import torch
import csv

def getRNA(filename, max_seq_length):

    if max_seq_length != 0:
       sequences = []
       with open(filename) as f:
            for line in f:
                line = line.rstrip('\n')
                if len(line) <= max_seq_length:
                   seq = []
                   diff = int(max_seq_length - len(line))
                   for char in line:
                       seq.append(str(char))
                   for i in range(len(seq), len(seq) + diff):
                       seq.append(0)
                   sequences.append(seq)
       return sequences

    elif max_seq_length == 0:
        with open(filename) as f:
            next(f)
            sequs = csv.reader(f)
            data = []
            for seq in sequs:
                data.append(seq)
            seq, dot = getData(data)

        return seq, np.array(dot)

def getData(data):
    sequences = []
    dotbrackets = []
    for i in range(len(data)-50000):

        sequences.append(data[i][0])

        bras = []

        for j in range(len(data[i][1])):

            if data[i][1][j] == '(':
                bras.append([1,0,0])

            elif data[i][1][j] == ')':
                bras.append([0,1,0])

            else:
                bras.append([0,0,1])

        bras = np.array(bras).reshape(30, 3)
        dotbrackets.append(bras)
    return sequences, dotbrackets

def getDotBrackets(filename, max_seq_length):
    bracket_reps = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) <= max_seq_length:
                rep = []
                diff = int(max_seq_length - len(line))
                for char in line:
                    if char == '(':
                        rep.append([1, 0, 0])
                    elif char == ')':
                        rep.append([0, 1, 0])
                    else:
                        rep.append([0, 0, 1])
                for i in range(len(rep), len(rep) + diff):
                    rep.append([0, 0, 0])
                rep = np.array(rep).reshape(max_seq_length, 3)
                bracket_reps.append(rep)
    return np.array(bracket_reps)


def getTrainingSets(sources, targets, max_seq_length, batch_size):

    if targets != 0:
        seqs = getRNA(sources, max_seq_length)
        dots = getDotBrackets(targets, max_seq_length)

        seqs = matrix.seqs2matrices(seqs)

        N = int(0.8* len(seqs))

        seqs = torch.from_numpy(seqs).float()
        dots = torch.from_numpy(dots).float()

        training_set = torch.utils.data.TensorDataset(seqs[0:N], dots[0:N])
        testing_set = torch.utils.data.TensorDataset(seqs[N+1:], dots[N+1:])

        train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    elif targets == 0:

        seqs, dots = getRNA(sources, 0)
        seqs = matrix.seqs2matrices(seqs)

        seqs = torch.from_numpy(seqs).float()
        dots = torch.from_numpy(dots).float()

        N = int(0.8*len(seqs))

        training_set = torch.utils.data.TensorDataset(seqs[0:N], dots[0:N])
        testing_set = torch.utils.data.TensorDataset(seqs[N+1:len(seqs)], dots[N+1:len(dots)])

        train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader
