''' 
Program to compile the text files into a usable dataset
'''
import os
import csv
data_path = '../../data/'

# Remove file if it exists
if os.path.exists(data_path+'scraper/dataset.csv'):
    os.remove(data_path+'scraper/dataset.csv')

with open(data_path+'scraper/dataset.csv', 'w+', newline='', encoding="utf-8") as csvfile:
    
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    
    for filename in os.listdir(data_path+'scraper/scraped_data/'):
        
        print(filename)
        
        # Annoyingly the first few are ANSI, not utf8
        x = filename.split('_')
        if int(x[0])<=22 and x[1] == '11':
            with open(data_path + 'scraper/scraped_data/' + filename, 'r') as f:
                data = f.read()
                split_data = data.split('\n')
                for d in split_data:
                    if any(c.isalnum() for c in d):
                        writer.writerow(['0', d.split(), x[3].split('.')[0]])
        else:
            with open(data_path + 'scraper/scraped_data/' + filename, 'r', encoding="utf-8") as f:
                data = f.read()
                split_data = data.split('\n')
                for d in split_data:
                    if any(c.isalnum() for c in d):
                        writer.writerow(['0', d.strip(), x[3].split('.')[0]])