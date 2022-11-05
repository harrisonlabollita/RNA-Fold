# The idea here is to build the vocabulary for the attention model. 
# We will need to one hot-encode our sequences and one-hot decode our sequences
# to properly interpret the results

def encode(sequence, max_length):
    # Input: A sequence that needs to be one hot encoded, max_length is the amount of padding that needs to be done for that sequence. 
    # Output: The one-hot encoded version of the sequence
    if len(sequence) != max_length:
       while len(sequence) < max_length:
	     sequence.append(0)       # pad the sequence with 0's so that it is the same length as the maximum length vector
    
    for i in range(len(sequence)):
	if sequence[i] == 'A':
	   # encode A
	elif sequence[i] == 'G':
	   # encode G
	elif sequence[i] == 'C':
           # encode C
	else:
	   # encode U 

     return sequence
