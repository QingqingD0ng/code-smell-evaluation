import urllib.request
from pyquery import PyQuery as pq
import datetime
import pandas as pd
import urllib.error

def task_func(url):
    if not url:
        raise ValueError("URL cannot be empty")

    response = urllib.request.urlopen(url)
    html_content = response.read()
    doc = pq(html_content)

    data = []
    for anchor in doc('a'):
        text = anchor.text()
        href = anchor.attr('href')
        fetch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data.append({'text': text, 'href': href, 'fetch_time': fetch_time})

    return pd.DataFrame(data)

# Example usage:
# df = task_func('https://en.wikibooks.org/wiki/Main_Page')