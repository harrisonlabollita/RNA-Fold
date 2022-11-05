def dynamicNussinov(pred_structure):
    N = pred_structure


    N[i,j]= max([N[i+1, j] + pred_structure[i, 2],
         N[i, j-1] + pred_structure[j, 2],
         N[i + 1, j-1] + delta(i, j, pred_structure),
         max([ N[i,k] + N[k+1, j] for k in range(i, j)])
    ])
    return N

def delta(i,j, pred_structure):
    return max([pred_structure[i, 0] + pred_structure[j, 1],
                pred_structure[i, 2] + pred_structure[j, 2]])


            
def OPT(i,j, sequence):
    """ returns the score of the optimal pairing between indices i and j"""
    #base case: no pairs allowed when i and j are less than 4 bases apart
    if i >= j-min_loop_length:
        return 0
    else:
        #i and j can either be paired or not be paired, if not paired then the optimal score is OPT(i,j-1)
        unpaired = OPT(i, j-1, sequence)

        #check if j can be involved in a pairing with a position t
        pairing = [1 + OPT(i, t-1, sequence) + OPT(t+1, j-1, sequence) for t in range(i, j-4)\
                   if pair_check((sequence[t], sequence[j]))]
        if not pairing:
            pairing = [0]
        paired = max(pairing)

        return max(unpaired, paired)

def nussinov(structure):
    # Sequence = redicted Structure from CNN
    N = len(structure)
    DP = structure

    #fill the DP matrix
    for k in range(min_loop_length, N):
        for i in range(N-k):
            j = i + k
            DP[i][j] = OPT(i,j, sequence)

    #copy values to lower triangle to avoid null references
    for i in range(N):
        for j in range(0, i):
            DP[i][j] = DP[j][i]

    traceback(0,N-1, structure, DP, sequence)
    return write_structure(sequence, structure)
