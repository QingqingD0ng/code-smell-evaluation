import urllib.request

from bs4 import BeautifulSoup

import csv

import os


CSV_FILE_PATH ='scraped_data.csv'


def task_func(url):

    response = urllib.request.urlopen(url)

    html = response.read()

    soup = BeautifulSoup(html, 'html.parser')


    # Assuming we want to scrape all paragraph tags

    paragraphs = soup.find_all('p')


    # Check if CSV file already exists and write header if necessary

    if not os.path.isfile(CSV_FILE_PATH):

        with open(CSV_FILE_PATH, 'w', newline='') as file:

            writer = csv.writer(file)

            writer.writerow(['Text'])


    # Write data to CSV

    with open(CSV_FILE_PATH, 'a', newline='') as file:

        writer = csv.writer(file)

        for paragraph in paragraphs:

            writer.writerow([paragraph.text.strip()])


    return CSV_FILE_PATH