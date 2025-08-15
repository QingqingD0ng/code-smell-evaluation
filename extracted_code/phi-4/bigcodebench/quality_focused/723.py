import urllib.request
from bs4 import BeautifulSoup
import csv
import os

CSV_FILE_PATH ='scraped_data.csv'

def task_func(url):
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser')

    data = []
    for item in soup.find_all('tag'):  # Replace 'tag' with the actual tag you want to scrape
        data.append([element.text for element in item.find_all('element_tag')])  # Replace 'element_tag'

    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    else:
        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    return CSV_FILE_PATH