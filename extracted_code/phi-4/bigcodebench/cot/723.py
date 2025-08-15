import urllib.request
from bs4 import BeautifulSoup
import csv

CSV_FILE_PATH ='scraped_data.csv'

def task_func(url):
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')

    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in rows:
            cols = row.find_all(['th', 'td'])
            writer.writerow([col.text.strip() for col in cols])

    return CSV_FILE_PATH