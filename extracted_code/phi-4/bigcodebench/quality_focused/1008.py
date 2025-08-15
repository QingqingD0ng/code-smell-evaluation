import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def task_func(url, table_id):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise e

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', id=table_id)

    if table is None:
        raise ValueError("Table with the specified ID not found.")

    headers = []
    rows = []

    header_row = table.find('tr')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all('th')]

    for row in table.find_all('tr')[1:]:
        rows.append([td.get_text(strip=True) for td in row.find_all('td')])

    if not rows:
        return pd.DataFrame()

    data = StringIO("\n".join(["\t".join(row) for row in rows]))
    df = pd.read_csv(data, sep="\t", names=headers)

    return df