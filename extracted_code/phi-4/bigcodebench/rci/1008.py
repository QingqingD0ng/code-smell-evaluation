import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url, table_id):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', id=table_id)

    if table is None:
        raise ValueError("Table with the specified ID not found.")

    # Convert the HTML table directly to a DataFrame
    df = pd.read_html(str(table))[0]

    # Check if the DataFrame is empty (i.e., no rows)
    if df.empty:
        return pd.DataFrame()

    return df