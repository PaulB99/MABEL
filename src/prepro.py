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
def create_dataset(size, path_name, add_npov = False ,mode='detection'):
    
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
    
    # Add NPOV dataset examples if requested
    if add_npov:
    
        with open('../data/NPOV/5gram-edits-train.tsv', encoding='utf8') as f:
            train_tsv = csv.reader(f, delimiter="\t")
            for row in train_tsv:
                if row[2] == 'true' or row[3] == 'true':
                    train.append((1,row[-2]))
                else:
                    train.append((0,row[-2]))
         
        with open('../data/NPOV/5gram-edits-dev.tsv', encoding='utf8') as f:    
            valid_tsv = csv.reader(f, delimiter="\t")
            for row in valid_tsv:
                if row[2] == 'true' or row[3] == 'true':
                    valid.append((1,row[-2]))
                else:
                    valid.append((0,row[-2]))
        
        with open('../data/NPOV/5gram-edits-test.tsv', encoding='utf8') as f:
            test_tsv = csv.reader(f, delimiter="\t")
            for row in test_tsv:
                if row[2] == 'true' or row[3] == 'true':
                    test.append((1,row[-2]))
                else:
                    test.append((0,row[-2]))
            
        random.shuffle(train)
        random.shuffle(valid)
        random.shuffle(test)
    
    # Make sure sequences aren't too long
    limit = 512
    for x in train:
        if len(x) >limit:
            train.remove(x)
    for x in valid:
        if len(x) >limit:
            valid.remove(x)
    for x in test:
        if len(x) >limit:
            test.remove(x)
    
    # Write to file
    if os.path.exists("../data/datasets/"+path_name+"/train_" + mode + ".csv"):
        os.remove("../data/datasets/"+path_name+"/train_" + mode + ".csv")
        
    if os.path.exists("../data/datasets/"+path_name+"/test_" + mode + ".csv"):
        os.remove("../data/datasets/"+path_name+"/test_" + mode + ".csv")
        
    if os.path.exists("../data/datasets/"+path_name+"/valid_" + mode +  ".csv"):
        os.remove("../data/datasets/"+path_name+"/valid_" + mode + ".csv")
        
    with open("../data/datasets/"+path_name+"/train_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('label', 'text'))
        for d in train:
            writer.writerow(d)
    with open("../data/datasets/"+path_name+"/test_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('label', 'text'))
        for d in test:
            writer.writerow(d)
            
    with open("../data/datasets/"+path_name+"/valid_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('label', 'text'))
        for d in valid:
            writer.writerow(d)
            
    print("Done!")
    
def create_seq2seq(size, path_name, mode='neutralisation', add_npov=False):
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
    
    biased = []
    
    # Read files
    with open('../data/WNC/biased.full', encoding = 'utf8') as f:
        for line in f:
            biased.append(line)
            
    # Randomise 
    random.shuffle(biased)
    
    for b in range(len(biased)):
        x = biased[b].split('	')
        if b < train_size:
            train.append((x[3],x[4])) 
        elif b < (train_size+valid_size):
            valid.append((x[3],x[4])) 
        else:
            test.append((x[3],x[4])) 
            
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    
    # Add NPOV dataset examples if requested
    if add_npov:
    
        with open('../data/NPOV/5gram-edits-train.tsv', encoding='utf8') as f:
            train_tsv = csv.reader(f, delimiter="\t")
            for row in train_tsv:
                if row[2] == 'true' or row[3] == 'true':
                    train.append((row[-2],row[-1]))
         
        with open('../data/NPOV/5gram-edits-dev.tsv', encoding='utf8') as f:    
            valid_tsv = csv.reader(f, delimiter="\t")
            for row in valid_tsv:
                if row[2] == 'true' or row[3] == 'true':
                    valid.append((row[-2],row[-1]))
        
        with open('../data/NPOV/5gram-edits-test.tsv', encoding='utf8') as f:
            test_tsv = csv.reader(f, delimiter="\t")
            for row in test_tsv:
                if row[2] == 'true' or row[3] == 'true':
                    test.append((row[-2],row[-1]))
            
        random.shuffle(train)
        random.shuffle(valid)
        random.shuffle(test)
        
    # Make sure sequences aren't too long
    limit = 512
    for x in train:
        if len(x) >limit:
            train.remove(x)
    for x in valid:
        if len(x) >limit:
            valid.remove(x)
    for x in test:
        if len(x) >limit:
            test.remove(x)
    
    # Write to file
    if os.path.exists("../data/datasets/"+path_name+ "/train_" + mode + ".csv"):
        os.remove("../data/datasets/"+path_name+"/train_" + mode + ".csv")
        
    if os.path.exists("../data/datasets/"+path_name+"/test_" + mode + ".csv"):
        os.remove("../data/datasets/"+path_name+"/test_" + mode + ".csv")
        
    if os.path.exists("../data/datasets/"+path_name+"/valid_" + mode +  ".csv"):
        os.remove("../data/datasets/"+path_name+"/valid_" + mode + ".csv")
        
    with open("../data/datasets/"+path_name+"/train_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('text', 'target'))
        for d in train:
            writer.writerow(d)
    with open("../data/datasets/"+path_name+"/test_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('text', 'target'))
        for d in test:
            writer.writerow(d)
            
    with open("../data/datasets/"+path_name+"/valid_" + mode + ".csv", 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.writer(csvfile) # TODO: Might have to change delimiter to avoid issues
        # Write column titles
        writer.writerow(('text', 'target'))
        for d in valid:
            writer.writerow(d)
            
    print('Done!')
    
create_dataset(200000, 'main', add_npov=False)

create_seq2seq(200000, 'main', add_npov=False)
    
    