import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def task_func(url, table_id):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', id=table_id)

    if table is None:
        raise ValueError("Table with the specified ID not found.")

    rows = table.find_all('tr')
    if not rows:
        return pd.DataFrame()

    headers = [th.get_text() for th in rows[0].find_all('th')]
    data = []

    for row in rows[1:]:
        cols = row.find_all('td')
        if cols:
            data.append([col.get_text() for col in cols])

    csv_data = StringIO()
    csv_writer = csv_data.write('\n'.join([','.join(headers)] + [','.join(row) for row in data]))

    csv_data.seek(0)
    df = pd.read_csv(csv_data)

    return df