import numpy as np

def pairs(a, b, x):
    if a == 'A' and b == 'U' or a == 'U' and b == 'A':
        return 2.0
    elif a == 'G' and b == 'C' or a == 'C' and b == 'G':
        return 3.0
    elif a == 'U' and b == 'G' or a == 'G' and b == 'U':
        return x
    else:
        return 0

def RNAmatrix(seq):
    N = len(seq)
    rnaMatrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            weight = 0
            rnaMatrix[i][j] = pairs(seq[i], seq[j], 0.8)
            if rnaMatrix[i][j] > 0:
                alpha = 0
                while i - alpha >= 0 and j + alpha <N:
                    P = pairs(seq[i -alpha], seq[j + alpha], 0.8)
                    if P == 0:
                        break
                    else:
                        weight += np.exp(-0.5 * alpha * alpha) * P
                        alpha +=1
            if weight > 0:
                beta = 1
                while i + beta < N and j - beta >=0:
                    P = pairs(seq[i + beta], seq[j - beta], 0.8)
                    if P == 0:
                        break
                    else:
                        weight += np.exp(-0.5 * beta * beta) * P
                        beta +=1
            rnaMatrix[i][j] = weight
    rnaMatrix = np.array(rnaMatrix)
    rnaMatrix /= np.amax(rnaMatrix)
    return rnaMatrix

def seqs2matrices(sequences):
    matrices = []
    for seq in sequences:
        matrices.append(RNAmatrix(seq))
    matrices = np.array(matrices)
    return matrices
