'''
Functions to preprocess the WNC into formats usable by my models
'''

import csv
import os
import random

# Create a small dataset for training and testing
def minidataset(name, size):
    # Variables
    dataset_size = size
    neutral = []
    biased = []
    
    # Read files
    with open('../data/WNC/neutral', encoding = 'utf8') as f:
        for line in f:
            neutral.append(line)
            
    with open('../data/WNC/biased.full', encoding = 'utf8') as f:
        for line in f:
            biased.append(line)
    
    # Randomise 
    random.shuffle(neutral)
    random.shuffle(biased)
    
    # Get correct size
    neutral2 = []
    biased2 = []
    for i in range(int(dataset_size/2)):
        neutral2.append(neutral[i])
    for i in range(int(dataset_size/2)):
        biased2.append(biased[i])
    
    # Process data 
    dataset = []
    for n in neutral2:
        x = n.split('	')
        dataset.append((0,x[-2])) # Skip out the id at the start
    for b in biased2:
        x = b.split('	')
        dataset.append((1,x[-4])) 
        
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Write to file
    if os.path.exists("../data/datasets/main/" + name + ".csv"):
        os.remove("../data/datasets/main/" + name + ".csv")
    with open("../data/datasets/main/" + name + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('label', 'text'))
        for d in dataset:
            writer.writerow(d)
            
    print("Done!")
    
minidataset('train', 100000)
minidataset('validate', 20000)
minidataset('test', 100000)
    