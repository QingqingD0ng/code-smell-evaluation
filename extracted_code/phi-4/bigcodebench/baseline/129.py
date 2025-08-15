import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url='http://example.com'):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError("Failed to connect to the URL.") from e

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')

    if not table:
        raise ValueError("No table found on the page.")

    headers = []
    for th in table.find_all('th'):
        headers.append(th.get_text(strip=True))
    
    rows = []
    for tr in table.find_all('tr'):
        cols = tr.find_all(['td', 'th'])
        cols = [col.get_text(strip=True) for col in cols]
        if cols:
            rows.append(cols)

    if not rows:
        raise ValueError("No table data found.")

    df = pd.DataFrame(rows, columns=headers if headers else None)
    return df