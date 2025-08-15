import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests

def extract_urls(myString):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, myString)

def fetch_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None

def get_page_title(content):
    soup = BeautifulSoup(content, 'html.parser')
    title_tag = soup.title
    return title_tag.string.strip() if title_tag and title_tag.string else None

def task_func(myString):
    urls = extract_urls(myString)
    
    if not urls:
        return "No valid URL found in the provided string."
    
    url = urls[0]
    content = fetch_url_content(url)
    
    if content is None:
        return f"Unable to fetch the content of the URL: {url}"
    
    title = get_page_title(content)
    
    if title:
        return title
    else:
        return "No title tag found in the webpage."