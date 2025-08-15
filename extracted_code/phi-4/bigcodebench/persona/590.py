import urllib.request
from pyquery import PyQuery as pq
from datetime import datetime
import pandas as pd
from urllib.error import URLError

def task_func(url):
    if not url:
        raise ValueError("The provided URL is invalid or empty.")
    
    try:
        with urllib.request.urlopen(url) as response:
            html_content = response.read()
    except URLError as e:
        raise URLError(f"Network issue or server error: {e}")

    document = pq(html_content)
    anchors = document('a')
    
    data = []
    for anchor in anchors.items():
        text = anchor.text()
        href = anchor.attr('href')
        data.append({'text': text, 'href': href})
    
    fetch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.DataFrame(data)
    df['fetch_time'] = fetch_time
    
    return df