import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url='http://example.com'):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException:
        raise ConnectionError("Failed to connect to the URL.")

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')

    if not table:
        raise ValueError("No table found on the page.")

    headers = []
    rows = []

    for row in table.find_all('tr'):
        cols = row.find_all(['th', 'td'])
        if not cols:
            continue

        if not headers:
            headers = [col.get_text(strip=True) for col in cols]
        else:
            rows.append([col.get_text(strip=True) for col in cols])

    if not rows:
        raise ValueError("No table data found.")

    return pd.DataFrame(rows, columns=headers)