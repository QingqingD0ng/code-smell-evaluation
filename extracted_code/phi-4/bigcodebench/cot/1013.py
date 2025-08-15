import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import csv

def task_func(url: str, base_url: str = "https://www.example.com", csv_file: str = "scraped_data.csv") -> int:
    response = requests.get(urljoin(base_url, url))
    soup = BeautifulSoup(response.content, 'html.parser')
    links = {urljoin(base_url, tag.get('href')) for tag in soup.find_all('a', href=True)}

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['URL'])
        for link in links:
            writer.writerow([link])

    return len(links)