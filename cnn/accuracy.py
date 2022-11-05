import numpy as np
import torch


def torch2num(output):
    new_ouput = output.detach().numpy()
    return new_ouput

def convert(output):
    for i in range(len(output)):
        for j in range(len(output[i])):

            idx = np.argmax(output[i][j])
            if idx == 0:
                output[i][j] =  np.array([1, 0, 0])
            elif idx == 1:
                output[i][j] = np.array([0, 1, 0])
            else:
                output[i][j] = np.array([0, 0, 1])

    return np.array(output)

def compute_acc(output, real):

    length = output.size()[0]
    rows = output.size()[1]
    cols = output.size()[2]
    output = torch2num(output)
    real = torch2num(real)
    output = convert(output)

    acc_list = []

    for i in range(length):
        count = 0

        for j in range(rows):
            for k in range(cols):
                if output[i][j][k] == real[i][j][k]:
                    count += 1

        acc_list.append(count/(rows*cols))

    return acc_list
