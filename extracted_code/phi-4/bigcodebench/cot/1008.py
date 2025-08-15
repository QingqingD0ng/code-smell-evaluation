import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url, table_id):
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', id=table_id)
    
    if not table:
        raise ValueError("Table with the specified ID not found.")
    
    rows = table.find_all('tr')
    if not rows or len(rows) < 2:
        return pd.DataFrame()
    
    headers = [th.get_text(strip=True) for th in rows[0].find_all('th')]
    data = [
        [td.get_text(strip=True) for td in row.find_all('td')]
        for row in rows[1:]
    ]
    
    return pd.DataFrame(data, columns=headers)