'''
Scraping data from various news sources
'''
import requests
import csv
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date
import os
import random
from selenium import webdriver
import time

reference_path = '../../data/scraper/News.csv'
data_path = '../../data/scraper/scraped_data/'
sources = []
with open(reference_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        sources.append(row)
        
today = date.today()
now = today.strftime("%d_%m_%Y")

for s in sources:
    URL = s[1]
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    filename = data_path + now
    
    # As each website is structured differently I need a different approach for each
    if(s[0] == 'BBC News'):
        headline_soup = soup.find_all('h3', class_='media__title')
        tagline_soup = soup.find_all('p', class_='media__summary')
        filename += '_bbc.txt'
    elif(s[0] == 'El Pais'):
        headline_soup = soup.find_all('h2')
        tagline_soup = soup.find_all('p', class_='c_d description | color_gray_medium block width_full false false')
        filename += '_ep.txt'
    elif(s[0] == 'Wall Street Journal'):
        driver = webdriver.Firefox() # Doesn't want to connect otherwise
        driver.get(URL)
        driver.execute_script("window.scrollTo(0, 10000)") 
        soup1 = BeautifulSoup(driver.page_source, 'html.parser')
        driver.close()
        headline_soup = soup1.find_all('h3')
        tagline_soup = soup1.find_all('p')
        filename += '_wsj.txt'
    elif(s[0] == 'Daily Mail'):
        headline_soup = soup.find_all('h2', class_='linkro-darkred')
        tagline_soup = soup.find_all('p')
        filename += '_dm.txt'
    elif(s[0] == 'Fox News'):
        headline_soup = soup.find_all('h2', class_='title title-color-default')
        tagline_soup = soup.find_all('p', class_='media__summary') # THERE AREN'T ANY!
        filename += '_fox.txt'
    elif(s[0] == 'New York Times'):
        headline_soup = soup.find_all('h3')
        tagline_soup = soup.find_all('p')
        filename += '_nyt.txt'
    elif(s[0] == 'Xinhua'):
        headline_soup = soup.find_all('h2') # TODO: Remove single words
        tagline_soup = soup.find_all('p')
        filename += '_xin.txt'
    elif(s[0] == 'Reuters'):
        headline_soup = soup.find_all(['h3', 'h2'], class_='story-title')
        tagline_soup = soup.find_all('p')
        filename += '_reu.txt'
    elif(s[0] == 'Russia Today'):
        headline_soup = soup.find_all('div', class_=['article-card__title article-card__title--size-18', 'main-promobox__heading'])
        tagline_soup = soup.find_all('p', class_='media__summary') # NONE!
        filename += '_rt.txt'
    elif(s[0] == 'The Guardian'):
        headline_soup = soup.find_all('span', class_= 'js-headline-text')
        tagline_soup = soup.find_all('div', class_='fc-item__standfirst')
        filename += '_gdn.txt'
    elif(s[0] == 'Times of India'):
        driver = webdriver.Firefox() # They want to be a pain with loading stuff via javascript
        driver.get(URL)
        driver.execute_script("window.scrollTo(0, 10000)") 
        time.sleep(5) # Give it time to catch up
        soup1 = BeautifulSoup(driver.page_source, 'html.parser')
        driver.close()
        headline_soup = soup1.find_all('a', class_='card-title')
        tagline_soup = soup1.find_all('p', class_='media__summary') # NONE!
        filename += '_ti.txt'
    elif(s[0] == 'Washington Post'):
        headline_soup = soup.find_all('h2', class_=['font--headline font-size-lg font-bold left relative', 'font--headline font-size-xs font-light left relative'])
        tagline_soup = soup.find_all('div', class_='bb pb-xs font--subhead 1h-fronts-sm font-light gray-dark')
        filename += '_wp.txt'
        
    headlines = []
    taglines = []
    
    for h in headline_soup:
        text = h.get_text()
        if (len(text) > 15) and not 'Reuters' in text: # Filter out actual titles and such (and Reuters annoyingly put their little blurb in the same element)
            headlines.append(text)
    for t in tagline_soup:
        text = t.get_text()
        if (len(text) > 15)  and not 'Reuters' in text: # Filter out actual titles and such 
            taglines.append(text)
    
    # Shuffle the headlines to get a good mix
    random.shuffle(headlines)
    random.shuffle(taglines)
    
    if os.path.exists(filename):
        os.remove(filename)
    
    f= open(filename,"w+")
    head_num = 15
    tag_num = 15
    
    print(s[0])
    print(len(headlines))
    print(len(taglines))
    
    if(len(taglines) < tag_num): # If there's not enough taglines, use more headlines
        diff = tag_num - len(taglines)
        tag_num = len(taglines)
        head_num+=diff
    
    if(len(headlines) < head_num): # If there's not enough headlines, use more taglines
        diff = head_num - len(headlines)
        head_num = len(headlines)
        if(tag_num == 15):
            tag_num+=diff
        else:
            print("Warning - Only " + str(head_num) + " headlines and " + str(tag_num) + " taglines available for " + s[0])
            
    if head_num > 0:   
        for h in range(head_num):
            f.write(headlines[h] + '\n')
    if tag_num > 0:
        for t in range(tag_num):
            f.write(taglines[t] + '\n')
    f.close()
    