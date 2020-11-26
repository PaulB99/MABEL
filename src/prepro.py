'''
Functions to preprocess the WNC into formats usable by my models
'''

import csv
import os

# Create a small dataset for training and testing
def minidataset():
    # Variables
    dataset_size = 3000
    neutral = []
    biased = []
    
    # Read files
    with open('../data/WNC/neutral', encoding = 'utf8') as f:
        i = 0
        for line in f:
            if i>=(dataset_size *(2/3)):
                break
            else:
                neutral.append(line)
                i+=1
            
    with open('../data/WNC/biased.full', encoding = 'utf8') as f:
        i = 0
        for line in f:
            if i>=(dataset_size *(1/3)):
                break
            else:
                biased.append(line)
                i+=1
    
    # Process data 
    dataset = []
    for n in neutral:
        x = n.split('	')
        dataset.append((0,x[-2])) # Skip out the id at the start
    for b in biased:
        x = b.split('	')
        dataset.append((1,x[-4])) 
        
    # Write to file
    if os.path.exists("../data/datasets/mini_dataset.csv"):
        os.remove("../data/datasets/mini_dataset.csv")
    with open("../data/datasets/mini_dataset.csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        for d in dataset:
            writer.writerow(d)
            
    print("Done!")
    
minidataset()
    