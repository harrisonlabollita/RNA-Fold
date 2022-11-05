# Convert the data files form the CT format to DOT format.
# Remove 5% of the sequences for testing
# Make text files containing the training data


import numpy as np
import glob, sys, os
import time



####################################################################################################################################################################
#  ATTENTION DO NOT RUN THIS CODE!                                                                                                                                #
#  The commited code was intended to run once. The purpose of this code is to remove 2200 sequences from our main sequences to reserve for testing our m           #
#  machine learning architectures at the very end of the process.                                                                                                  #
#                                                                                                                                                                  #
#                                                                                                                                                                  #
# seq_files = os.listdir('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/seed_structures/') #
# dir = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/seed_structures/'                   #
# new_dir = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/doNotTouch/'                       #                                                                                                                                                   #
# condition = 2200                                                                                                                                                 #
# iter = 0                                                                                                                                                         #
# for file in seq_files:                                                                                                                                           #
#    name = file                                                                                                                                                   #
#    if iter < condition:                                                                                                                                          #
#        x = np.random.random()                                                                                                                                    #
#        if x < 0.3:                                                                                                                                               #
#            os.rename(dir + name, new_dir + name)                                                                                                                 #
#            iter += 1                                                                                                                                             #
# print('Removed %d files from seed_structures to doNotTouch!' %(iter))                                                                                            #
####################################################################################################################################################################


def str_append(s):
    output = ''
    output += s
    return output

def readFile(file):
    record = {}
    dots = ''
    with open(file, "r") as f:
        for line in f.readlines()[1:]:
            print(line)
            l = [x for x in line.split('') if x != '']
            print(l[0])
            n = int(l[0])
            base = l[1]
            partner = int(l[4])
            record[n] = [base, partner]
    seq = "".join([record[x][0] for x in sorted(record.keys())])
    for x in sorted(record.keys()):
        val = record[x][1]
        if val > x:
            s = '('
            dots += str_append(s)
        elif val == 0:
            s = '.'
            dots += str_append(s)
        elif val < x:
            s = ')'
            dots += str_append(s)
    return seq, dots

seq_files = glob.glob('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/seed_structures/*')

sequences = []
dotbrackets = []
bonds = []
strandLengths = []

start = time.time()
for file in seq_files:

    seq, dots = readFile(file)  # Read in file with sequence
    sequences.append(seq)       # append the sequence to an array
    dotbrackets.append(dots)    # append the dotBracket rep to an array
    strandLengths.append(len(seq))  # append the length of that sequence to an array

    bond = 0
    for i in range(len(dots)):
        if dots[i] != '.':
            bond += 1

    bond /= 2.0
    bonds.append(bond)


strandLengths = np.array(strandLengths)
end = time.time()

print('Processed %d strands in %0.2f seconds' %(len(strandLengths), end - start))
print("Shortest strand:", min(strandLengths))
print("Longest strand:", max(strandLengths))
print("Strands with length < 1000:", len(np.where(strandLengths[:]<1000)[0]))
print("Length mean:", np.mean(strandLengths))
print("Length std:", np.std(strandLengths))
print("Average bonds:", np.mean(bonds))

with open('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/sequences.txt', 'w') as f:
    for seq in sequences:
        f.write('%s\n' %(seq))
f.close()

with open('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/dotbrackets.txt', 'w') as f:
    for dot in dotbrackets:
        f.write('%s\n' %(dot))
f.close()
