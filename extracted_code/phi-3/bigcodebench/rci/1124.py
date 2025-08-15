import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests

def extract_url_title(input_string):
    urls = re.findall(r'https?://\S+', input_string)
    if not urls:
        return "No valid URL found in the provided string."

    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code!= 200:
                return f"Unable to fetch the content of the URL: {url}"
            soup = BeautifulSoup(response.content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.text.strip()
            return "No title tag found in the webpage."
        except requests.RequestException:
            return f"Unable to fetch the content of the URL: {url}"

# Example usage:
# print(extract_url_title('Check this out: https://www.google.com'))
# print(extract_url_title('No URL here'))
# print(extract_url_title('Check this broken link: https://www.thisdoesnotexist12345.com'))