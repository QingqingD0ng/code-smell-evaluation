import requests
from lxml import html
import pandas as pd
import sqlite3
from urllib.parse import urlparse

def task_func(webpage_url: str, database_name: str = "my_database.db", table_name: str = "my_table") -> int:
    try:
        # Validate URL or file path
        parsed_url = urlparse(webpage_url)
        if not parsed_url.scheme and not webpage_url.startswith("file://"):
            raise ValueError("Invalid URL or file path")

        # Fetch content
        if webpage_url.startswith("file://"):
            with open(webpage_url[7:], 'r') as file:
                content = file.read()
        else:
            response = requests.get(webpage_url, timeout=10)
            response.raise_for_status()
            content = response.text

        # Parse HTML
        tree = html.fromstring(content)
        tables = tree.xpath('//table')
        if not tables:
            return 0

        df = pd.read_html(html.tostring(tables[0]), flavor='lxml')[0]
        if df.empty:
            return 0

        # Store in SQLite
        with sqlite3.connect(database_name) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)

        return len(df)

    except (requests.RequestException, sqlite3.DatabaseError, ValueError) as e:
        raise e