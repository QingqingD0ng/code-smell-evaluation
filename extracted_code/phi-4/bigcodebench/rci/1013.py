import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import csv
from requests.exceptions import RequestException

def task_func(url: str, base_url: str = "https://www.example.com", csv_file: str = "scraped_data.csv") -> int:
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        full_url = urljoin(base_url, url)
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
    except RequestException:
        return 0

    soup = BeautifulSoup(response.text, 'html.parser')
    links = set()

    for link in soup.find_all('a', href=True):
        absolute_link = urljoin(base_url, link['href'])
        links.add(absolute_link)

    try:
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['URL'])
            for link in links:
                writer.writerow([link])
    except IOError:
        return 0

    return len(links)