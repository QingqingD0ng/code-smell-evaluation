import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url='http://example.com'):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(f"Error connecting to {url}: {e}")

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    
    if not table:
        raise ValueError("No table found on the page.")

    headers = []
    rows = []

    # Extract headers
    header_row = table.find('tr')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

    # Extract rows
    for row in table.find_all('tr'):
        cols = row.find_all(['td', 'th'])
        if cols:
            rows.append([col.get_text(strip=True) for col in cols])

    if not rows:
        raise ValueError("No data found in the table.")

    df = pd.DataFrame(rows[1:], columns=headers) if headers else pd.DataFrame(rows[1:], columns=None)

    return df