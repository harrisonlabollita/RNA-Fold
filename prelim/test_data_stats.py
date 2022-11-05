import numpy as np
import matplotlib.pyplot as plt
import time

# Write function to convert the testing files directory to a single txt file for manipulation

sequences = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/sequences.txt'


def readFile(file):
    seqs = []
    with open(file, 'r') as f:
        for line in f.readlines():
            seqs.append(line)
    seqs = np.array(seqs)
    return seqs


start = time.time()
# Training RNA Sequences
strands = readFile(sequences)
dots = readFile(dotbras)
bonds = bondCount(dots)

# Lengths of all of the sequences
lengths = [ len(strands[i]) for i in range(len(strands))]
lengths =np.array(lengths)
end = time.time()

print('Finished processing training set in %0.2f seconds' %(end - start))


print('|--------------------------|')
print('|   RNA Testing Dataset   |')
print('|--------------------------|')
print('|  Total      |    %d   |' %(len(lengths)))
print('|  Mean       |    %0.2f  |' %(np.mean(lengths)))
print('|  StDev      |    %0.2f  |' %(np.std(lengths)))
print('|  Longest    |    %d    |' %(max(lengths)))
print('|  Shortest   |    %d      |' %(min(lengths)))
print('|  Len < 1000 |    %d   |' %(len(np.where(lengths[:]<1000)[0])))
print('|  Avg. Bonds |    %0.1f    |' %(np.mean(bonds)))
print('|--------------------------|')




plt.figure()
plt.grid(True, linestyle = ':', linewidth = 1)
plt.hist(lengths, bins = 'auto', histtype = 'bar', rwidth = 0.9, color = 'black', alpha = 0.9, linewidth = 2)
plt.title('RNA Testing Dataset')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()
