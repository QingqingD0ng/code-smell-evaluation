import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import csv

def task_func(url: str, base_url: str = "https://www.example.com") -> (str, int):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        return ("", 0)

    soup = BeautifulSoup(response.text, 'html.parser')
    links = set()
    for link in soup.find_all('a', href=True):
        abs_url = urljoin(base_url, link['href'])
        if abs_url.startswith('http'):
            links.add(abs_url)

    csv_filename ='scraped_data.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for link in links:
            writer.writerow([link])
    return (csv_filename, len(links))