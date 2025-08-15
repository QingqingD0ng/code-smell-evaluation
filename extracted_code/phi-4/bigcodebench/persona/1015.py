import requests
from lxml import html
import pandas as pd
import sqlite3

def task_func(webpage_url: str, database_name: str = "my_database.db") -> int:
    try:
        # Fetch the HTML content from the URL or local file
        if webpage_url.startswith("file://"):
            webpage_url = webpage_url[7:]  # Remove the 'file://' prefix
            with open(webpage_url, 'r') as file:
                page_content = file.read()
        else:
            response = requests.get(webpage_url)
            response.raise_for_status()
            page_content = response.text

        # Parse the HTML content
        tree = html.fromstring(page_content)
        table = tree.xpath('//table')[0]  # Assumes the first table is the target

        # Convert the table to a DataFrame
        df = pd.read_html(html.tostring(table).decode('utf-8'))[0]

        # Connect to the SQLite database
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()

        # Drop the existing table if it exists
        cursor.execute("DROP TABLE IF EXISTS my_table")

        # Create a new table and insert the DataFrame data
        df.to_sql('my_table', conn, index=False, if_exists='replace')

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

        # Return the number of rows in the DataFrame
        return len(df)

    except requests.RequestException as e:
        raise requests.RequestException("Network issue occurred while accessing the URL.") from e
    except sqlite3.DatabaseError as e:
        raise sqlite3.DatabaseError("Database error occurred.") from e
    except Exception as e:
        raise Exception("An error occurred while processing the HTML table.") from e