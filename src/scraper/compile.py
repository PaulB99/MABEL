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
    
    writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    for filename in os.listdir(data_path+'scraper/scraped_data/'):
        
        print(filename)
        
        # Annoyingly the first few are ANSI, not utf8
        x = filename.split('_')
        if int(x[0])<=22 and x[1] == '11':
            with open(data_path + 'scraper/scraped_data/' + filename, 'r') as f:
                data = f.read()
                writer.writerow(data)
        else:
            with open(data_path + 'scraper/scraped_data/' + filename, 'r', encoding="utf-8") as f:
                data = f.read()
                writer.writerow(data)