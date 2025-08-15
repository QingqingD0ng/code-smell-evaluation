import urllib.request
from bs4 import BeautifulSoup
import csv
import os

CSV_FILE_PATH ='scraped_data.csv'

def task_func(url):
    with urllib.request.urlopen(url) as response:
        html_content = response.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    data_to_save = []
    for element in soup.find_all('div', class_='data-class'):
        data_to_save.append(element.text.strip())
    with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Data'])
        for item in data_to_save:
            writer.writerow([item])
    return CSV_FILE_PATH