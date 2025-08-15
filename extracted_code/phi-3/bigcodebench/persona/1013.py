import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import csv


def task_func(url: str, base_url: str = "https://www.example.com", csv_file: str = "scraped_data.csv") -> int:
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        
        for a_tag in soup.find_all('a', href=True):
            absolute_url = urljoin(base_url, a_tag['href'])
            links.add(absolute_url)
        
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for link in links:
                writer.writerow([link])
        
        return len(links)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


# Example calls
print(task_func('/mywebpage'))
print(task_func('/anotherpage', base_url='https://www.different.com', csv_file='other_links.csv'))