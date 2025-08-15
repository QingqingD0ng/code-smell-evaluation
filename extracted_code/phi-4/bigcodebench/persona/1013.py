import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import csv

def task_func(
    url: str,
    base_url: str = "https://www.example.com",
    csv_file: str = "scraped_data.csv",
) -> int:
    try:
        full_url = urljoin(base_url, url)
        response = requests.get(full_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return 0

    soup = BeautifulSoup(response.content, 'html.parser')
    links = set()
    
    for link in soup.find_all('a', href=True):
        absolute_url = urljoin(base_url, link['href'])
        links.add(absolute_url)

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['URL'])
        for link in links:
            writer.writerow([link])

    return len(links)