import requests
from lxml import html
import pandas as pd
import sqlite3

def task_func(webpage_url: str, database_name: str = "my_database.db") -> int:
    response = requests.get(webpage_url)
    response.raise_for_status()
    tree = html.fromstring(response.content)
    table = tree.xpath('//table')
    
    if not table:
        return 0
    
    data = pd.read_html(html.tostring(table[0]))[0]
    
    if data.empty:
        return 0
    
    conn = sqlite3.connect(database_name)
    data.to_sql('my_table', conn, index=False, if_exists='replace')
    conn.close()
    
    return len(data)