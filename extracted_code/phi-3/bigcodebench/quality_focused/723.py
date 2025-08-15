import urllib.request
from bs4 import BeautifulSoup
import csv
import os

CSV_FILE_PATH ='scraped_data.csv'

def task_func(url):
    # Fetch the content from the URL
    with urllib.request.urlopen(url) as response:
        page_content = response.read()

    # Parse the content using BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    # Define the data structure for scraped data
    scraped_data = []

    # Extract the desired data from the soup object based on the HTML structure
    # Example: scraped_data.append({'title': soup.find('h1').text})

    # Save the scraped data to a CSV file
    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['field1', 'field2',...])
        writer.writeheader()
        for data in scraped_data:
            writer.writerow(data)

    return CSV_FILE_PATH

# Example usage
# task_func('http://www.example.com/')