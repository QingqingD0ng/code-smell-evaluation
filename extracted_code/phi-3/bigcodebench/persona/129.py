import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url='http://example.com'):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    if not table:
        raise ValueError('No table data found on the webpage.')
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = [[td.get_text(strip=True) for td in tr.find_all('td')] for tr in table.find_all('tr')]
    df = pd.DataFrame(rows, columns=headers)
    return df