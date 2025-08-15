import requests
from lxml import html
import pandas as pd
import sqlite3

def task_func(webpage_url: str, database_name: str = "my_database.db") -> int:
    try:
        # Fetch HTML content from URL
        response = requests.get(webpage_url)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        
        # Parse table from HTML
        df = pd.read_html(response.content, flavor='lxml')[0]
        
        # Drop empty rows from DataFrame
        df.dropna(how='all', inplace=True)
        
        # Connect to SQLite database
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()
        
        # Create table if it does not exist
        cursor.execute("DROP TABLE IF EXISTS my_table")
        df.to_sql("my_table", conn, if_exists="replace", index=False)
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        # Return number of rows in the table
        return len(df)
    except requests.RequestException as e:
        raise e
    except sqlite3.DatabaseError as e:
        raise e