# REINFORCEMENT LEARNING FOR RNA FOLDING PREDICTION
# --------------------------------------------------
# Input: RNA Sequence
# Convert sequence to matrix (following Zhang et. al.)
# Use CNN Model to predict a pairing map
#
#              ACTIONS
#         (         )        .
#     ----------------------------
#  S  |   0.1  |  0.4    |  0.5  |
#  E  |   0.9  |  0.1    |  0.0  |
#  Q  |   0.6  |  0.2    |  0.2  |
#  U  |   0.2  |  0.6    |  0.2  |
#  E  |   0.3  |  0.1    |  0.6  |
#  N  |   0.6  |  0.1    |  0.3  |
#  C  |   0.5  |  0.2    |  0.3  |
#  E  |   0.2  |  0.8    |  0.0  |
#     ----------------------------
# Use this pairing map for reinforcment learning (Q learning)
#      Q(s, a) <----- Q(s, a) + alpha ( r + gamma * max( Q(s', a'))),
# where r is the immediate reward, alpha is the learning rate, gamma is the
# discount, and max(Q(s', a')) is the maximum reaward for the next state.
# Here I envision each state as a nucleotide, which is corresponding actions associated with it, pair forward,
# pair back, or do not pair.


import numpy as np

def pair(A, B):
    if A == 'A' and B == 'U' or A == 'U' and B == 'A':
        return True
    elif A == 'G' and B == 'C' or A == 'C' and B == 'G':
        return True
    elif A == 'G' and B == 'C' or A == 'C' and B == 'G':
        return True
    else:
        return False

def init_pair_map(sequence):
    N = len(sequence)
    return np.zeros(N)

def update_pair_memory(pair_map, action):
    # action will be pair two nucleotides or not pair
    # action will be the pairs if
    if len(action) > 1:
        # is an array of two pairs
        pair_map[action[0]] = action[0]
        pair_map[action[1]] = action[1]
    else:
        pair_map[action[0]] = action[0]
    return pair_map

def find_possible_pairings(i, sequence):
    # find all of the physical pairings for this nucleotide
    # input: nucleotide, index of nucleotide, and the sequence
    nucl = sequence[i]

    scores = []
    if i == 0:
        # then we are at the beginning of the sequence so we need to explore all of the possibilities to the right:
        for j in range(len(sequence)):
            if j>= 4:
                if pair(nucl, sequence[j]):
                # keep track of this pair to be score
                    gamma = np.log(abs(j - i))
                    scores.append([j, gamma]) # append the position of the nucleotide and the cost of pairing with that nucleotide
    else:
        for j in range(i + 4, len(sequence)):
            # find the pairs after index i
            if pair(nucl, sequence[j]):
                gamma = np.log(abs(j - i))
                scores.append([j, gamma])
            # find the pairs before index i
        if i >= 5:
            for k in range(0, i-1):
                if pair(nucl, sequence[k]):
                    gamma = np.log(abs(k - i))
                    scores.append([k, gamma])
    return scores

def max_score(scores):
    max = scores[0]
    for i in rane(len(scores)):
        if scores[i][1] > max:
            max = scores[i]
    return max

def calculate_Q(i, sequence, pair_map, net):
    if i == len(sequence):
        # have reached the end of the sequence
        return
    lr = 0.01 # learning rate for the model
    # net is the network's prediction for the pairing of the bases
    # i is the nucleotid of interest or the state
    # sequence: we need the sequence to find the pairs for this matrix
    actions = np.array(net[i]) # contains the scores from the net on what action to choose
    scores = find_possible_pairings(i, sequence)
    if np.argmax(actions) == 0:
        choice_1 = max_score(scores) + lr * ( actions[0] + cost * calculate_Q(i+1, sequence, pair_map, net))
        # this says pair with some one to th right
    elif np.argmax(actions) == 1:
        # pair with some on the left
        choice_2 = max_score(scores) + lr * ( actions[1] + cost * calculate_Q(i+1, sequence, pair_map, net))
    else:
        # don't pair
        choice_3 = lr * ( actions[2] + cost * calculate_Q(i+1, sequence, pair_map, net))
    decision = [choice1, choice2, choice3]
    act = max(decision)
    return
