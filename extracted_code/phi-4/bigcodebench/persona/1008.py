import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def task_func(url, table_id):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', id=table_id)

    if table is None:
        raise ValueError("Table with the specified ID not found.")

    rows = table.find_all('tr')
    if not rows:
        return pd.DataFrame()

    data = []
    headers = [th.get_text(strip=True) for th in rows[0].find_all('th')]
    for row in rows[1:]:
        cols = row.find_all('td')
        if cols:
            data.append([col.get_text(strip=True) for col in cols])

    csv_data = StringIO()
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(csv_data, index=False)
    csv_data.seek(0)

    return pd.read_csv(csv_data)