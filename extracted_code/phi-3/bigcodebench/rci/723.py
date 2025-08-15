import urllib.request
from bs4 import BeautifulSoup
import csv

CSV_FILE_PATH ='scraped_data.csv'

def scrape_and_save_data(url):
    try:
        with urllib.request.urlopen(url) as response:
            html_content = response.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            data_to_save = [data.text.strip() for data in soup.find_all('div', class_='data-class')]
    except urllib.error.URLError as e:
        print(f'Failed to retrieve data from {url}. Error: {e}')
        return None
    except Exception as e:
        print(f'An error occurred while parsing the HTML content. Error: {e}')
        return None
    else:
        with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Data'])
            for item in data_to_save:
                writer.writerow([item])
        return CSV_FILE_PATH

# Usage example
if __name__ == '__main__':
    url = 'http://www.example.com/'
    scrape_and_save_data(url)