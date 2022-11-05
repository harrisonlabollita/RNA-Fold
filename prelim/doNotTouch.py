import numpy as np
import sys, os

sequencesFile = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/sequences.txt'
dotsFile = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/dotbrackets.txt'

def readFile(file):
    with open(file, 'r') as f:
        for line in f.readlines():
            print(line)

readFile(sequencesFile)
