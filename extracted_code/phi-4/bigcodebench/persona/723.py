import urllib.request
from bs4 import BeautifulSoup
import csv
import os

CSV_FILE_PATH ='scraped_data.csv'

def task_func(url):
    try:
        # Open the URL and read the content
        with urllib.request.urlopen(url) as response:
            html = response.read()

        # Parse the HTML content
        soup = BeautifulSoup(html, 'html.parser')

        # Example: Extracting all links from the webpage
        links = [(a.get_text(), a['href']) for a in soup.find_all('a', href=True)]

        # Ensure the CSV file directory exists
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

        # Write data to CSV
        with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Text', 'URL'])  # Header
            writer.writerows(links)

    except urllib.error.URLError as e:
        print(f"Failed to open URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return CSV_FILE_PATH