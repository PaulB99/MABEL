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
    
# Create a dataset of a given size
def create_dataset(size, mode='detection'):
    
    train = []
    valid = []
    test = []
    
    # The proportion to be assigned to each
    train_frac = 0.7
    valid_frac = 0.15
    test_frac = 0.15
    
    # The size of each part
    train_size = int(size*train_frac)
    valid_size = int(size*valid_frac)
    test_size = int(size*test_frac)
    
    # Ensure they're the right size
    
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
    for i in range(int(size/2)):
        neutral2.append(neutral[i])
    for i in range(int(size/2)):
        biased2.append(biased[i])
        
    # Process data 
    dataset = []
    for n in range(len(neutral2)):
        x = neutral2[n].split('	')
        if n < int(train_size/2):
            train.append((0,x[-2])) # Skip out the id at the start
        elif n < int((train_size+valid_size)/2):
            valid.append((0,x[-2]))
        else:
            test.append((0,x[-2]))
            
    for b in range(len(biased2)):
        x = biased2[b].split('	')
        if b < int(train_size/2):
            train.append((1,x[-4])) 
        elif b < int((train_size+valid_size)/2):
            valid.append((1,x[-4])) 
        else:
            test.append((1,x[-4])) 
      
    # Randomise
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    
    # Write to file
    if os.path.exists("../data/datasets/main/train_" + mode + ".csv"):
        os.remove("../data/datasets/main/train_" + mode + ".csv")
        
    if os.path.exists("../data/datasets/main/test_" + mode + ".csv"):
        os.remove("../data/datasets/main/test_" + mode + ".csv")
        
    if os.path.exists("../data/datasets/main/valid_" + mode +  ".csv"):
        os.remove("../data/datasets/main/valid_" + mode + ".csv")
        
    with open("../data/datasets/main/train_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('label', 'text'))
        for d in train:
            writer.writerow(d)
    with open("../data/datasets/main/test_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('label', 'text'))
        for d in test:
            writer.writerow(d)
            
    with open("../data/datasets/main/valid_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('label', 'text'))
        for d in valid:
            writer.writerow(d)
            
    print("Done!")
    
create_dataset(100000)
    
    