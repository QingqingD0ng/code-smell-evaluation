import urllib.request

from bs4 import BeautifulSoup

import csv

import os


CSV_FILE_PATH ='scraped_data.csv'


def task_func(url):
    request = urllib.request.urlopen(url)
    page_content = request.read()
    soup = BeautifulSoup(page_content, 'html.parser')

    with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Data Headers'])
        for element in soup.find_all('div'):
            writer.writerow([element.text])

    return CSV_FILE_PATH


if __name__ == '__main__':
    url_to_scrape = 'http://www.example.com/'
    csv_file_path = task_func(url_to_scrape)