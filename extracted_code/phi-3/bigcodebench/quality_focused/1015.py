import requests
from lxml import html
import pandas as pd
import sqlite3

def task_func(webpage_url: str, database_name: str = "my_database.db") -> int:
    try:
        response = requests.get(webpage_url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise e

    tree = html.fromstring(response.content)
    table_rows = tree.xpath('//table//tr')
    if not table_rows:
        return 0

    data = []
    for row in table_rows[1:]:  # Skip the header row
        cols = row.xpath('.//td/text()')
        data.append([col.strip() for col in cols])

    df = pd.DataFrame(data)
    df.to_sql('my_table', sqlite3.connect(database_name), if_exists='replace', index=False)

    return len(df)