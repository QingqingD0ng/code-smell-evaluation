import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def task_func(url, table_id):
    response = requests.get(url)
    response.raise_for_status()  # Raises HTTPError for unsuccessful HTTP requests
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id=table_id)
    if table is None:
        raise ValueError(f"Table with the specified ID '{table_id}' not found.")
    rows = table.find_all('tr')
    if not rows:
        return pd.DataFrame()  # Returns an empty DataFrame if no rows are found
    data = []
    headers = [th.text.strip() for th in rows[0].find_all('th')]  # Extract column headers
    for row in rows[1:]:
        cols = row.find_all('td')
        data.append([col.text.strip() for col in cols])
    df = pd.DataFrame(data, columns=headers)
    return df