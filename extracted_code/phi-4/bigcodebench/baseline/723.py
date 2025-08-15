import urllib.request
from bs4 import BeautifulSoup
import csv
import os

CSV_FILE_PATH ='scraped_data.csv'

def task_func(url):
    # Open the URL and read the page content
    response = urllib.request.urlopen(url)
    html_content = response.read()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract data (for example, all text within <p> tags)
    data = [p.text for p in soup.find_all('p')]

    # Ensure the directory exists
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

    # Write the extracted data to a CSV file
    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for item in data:
            writer.writerow([item])

    return CSV_FILE_PATH