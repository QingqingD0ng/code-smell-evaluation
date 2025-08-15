import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def task_func(url, table_id):
    response = requests.get(url)
    if response.status_code!= 200:
        raise requests.exceptions.HTTPError("Failed to retrieve webpage content.")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': table_id})
    if not table:
        raise ValueError("Table with the specified ID not found.")
    
    rows = table.find_all('tr')
    if not rows:
        return pd.DataFrame()
    
    data = {
        column.text.strip(): [cell.text.strip() for cell in row.find_all('td')]
        for column, row in zip(table.find_all('th'), rows[1:])
    }
    
    df = pd.DataFrame(data)
    return df