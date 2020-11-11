'''
Scraping data from various news sources
'''
import requests
import csv
import pandas as pd
from bs4 import BeautifulSoup

reference_path = '../../data/scraper/News.csv'
sources = []
with open(reference_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        sources.append(row)

for s in sources:
    URL = s[1]
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # As each website is structured differently I need a different approach for each
    if(s[0] == 'BBC News'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')    
    elif(s[0] == 'El Pais'):
        headline_soup = soup.find_all('h2', class_='c_h headline | color_gray_ultra_dark font_secondary width_full  headline_md ')
        tagline_soup = soup.find_all('p', class_='c_d description | color_gray_medium block width_full false false')
    elif(s[0] == 'CNN'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'Daily Mail'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'Fox News'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'New York Times'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'Xinhua'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'Reuters'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'Russia Today'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'The Guardian'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'Times of India'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
    elif(s[0] == 'Washington Post'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
        
    headlines = []
    taglines = []
    
    for h in headline_soup:
        text = h.get_text()
        headlines.append(h)
    for t in tagline_soup:
        text = t.get_text()
        taglines.append(t)
        
        
        
        
        