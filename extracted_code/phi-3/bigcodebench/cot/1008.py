import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def task_func(url, table_id):
    response = requests.get(url)
    if response.status_code!= 200:
        raise requests.exceptions.HTTPError
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id=table_id)
    if not table:
        raise ValueError(f"Table with the specified ID '{table_id}' not found.")
    
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = table.find_all('tr')[1:]
    
    if not rows:
        return pd.DataFrame(columns=headers)
    
    data = [
        {headers[i]: cell.get_text(strip=True) for i, cell in enumerate(row.find_all(['th', 'td']))}
        for row in rows
    ]
    
    df = pd.DataFrame(data)
    
    return df