import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url='http://example.com'):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')

    if table is None:
        raise ValueError("No table found on the page.")

    headers = []
    for header in table.find_all('th'):
        headers.append(header.get_text(strip=True))

    rows = []
    for row in table.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        if cells:
            rows.append(cells)

    if not rows:
        raise ValueError("No data found in the table.")

    df = pd.DataFrame(rows[1:], columns=headers[:len(rows[0])])
    return df