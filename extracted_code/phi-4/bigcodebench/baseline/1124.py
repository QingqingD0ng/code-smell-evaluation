import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests

def task_func(myString):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, myString)
    
    if not urls:
        return "No valid URL found in the provided string."
    
    url = urls[0]
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException:
        return f"Unable to fetch the content of the URL: {url}"
    
    soup = BeautifulSoup(response.content, 'html.parser')
    title_tag = soup.title
    
    if title_tag:
        return title_tag.string.strip()
    else:
        return "No title tag found in the webpage."