import requests

from urllib.parse import urljoin

from bs4 import BeautifulSoup

import csv

import os


def task_func(url: str, base_url: str = "https://www.example.com", csv_file: str = "scraped_data.csv") -> int:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                links.add(absolute_url)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for link in links:
                writer.writerow([link])
        return len(links)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


# Example usage without docstrings or comments:

url_to_scrape = '/mywebpage'

base_url_to_use = 'https://www.example.com'

csv_filename_to_create ='scraped_data.csv'

unique_links_count = task_func(url_to_scrape, base_url_to_use, csv_filename_to_create)

print(f"Number of unique absolute links: {unique_links_count}")