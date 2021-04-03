''' 
Program to compile the text files into one file
'''
import os
import csv
data_path = '../../data/'

# Remove file if it exists
if os.path.exists(data_path+'scraper/dataset.txt'):
    os.remove(data_path+'scraper/dataset.txt')

with open(data_path+'scraper/dataset.txt', 'w+', newline='', encoding="utf-8") as outfile:
    
    for filename in os.listdir(data_path+'scraper/scraped_data/'):
        
        print(filename)
        
        # Annoyingly the first few are ANSI, not utf8
        x = filename.split('_')            
        if int(x[0])<=22 and x[1] == '11':
            with open(data_path + 'scraper/scraped_data/' + filename, 'r') as infile:
                outfile.write(infile.read())

        else:
            with open(data_path + 'scraper/scraped_data/' + filename, 'r', encoding="utf-8") as infile:
                outfile.write(infile.read())
