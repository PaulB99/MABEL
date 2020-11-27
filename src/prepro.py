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
    if os.path.exists("../data/datasets/mini/" + name + ".csv"):
        os.remove("../data/datasets/mini/" + name + ".csv")
    with open("../data/datasets/mini/" + name + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        for d in dataset:
            writer.writerow(d)
            
    print("Done!")
    
minidataset('train', 3000)
minidataset('validate', 1000)
    