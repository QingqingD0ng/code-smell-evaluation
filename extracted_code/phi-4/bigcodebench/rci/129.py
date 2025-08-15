import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Optional

def extract_text(elements):
    return [element.get_text(strip=True) for element in elements]

def task_func(url: str = 'http://example.com') -> Optional[pd.DataFrame]:
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        raise ConnectionError("Failed to connect to the URL.")

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')

    if not table:
        raise ValueError("No table found on the page.")

    headers = extract_text(table.find_all('th'))
    rows = [extract_text(tr.find_all(['td', 'th'])) for tr in table.find_all('tr') if tr.find_all(['td', 'th'])]

    if not rows:
        raise ValueError("No table data found.")

    df = pd.DataFrame(rows, columns=headers if headers else None)
    return df