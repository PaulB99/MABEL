'''
Scraping data from various news sources
'''
import requests
import csv
import pandas as pd

reference_path = '../../data/scraper/News.csv'
sources = []
with open(reference_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        sources.append(row)

for s in sources:
    URL = s[1]
    page = requests.get(URL)