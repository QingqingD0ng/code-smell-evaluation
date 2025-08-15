import bs4
import requests
import re
import json

def task_func(url: str, output_path: str) -> list:
    phone_numbers = []

    if url.startswith('file://'):
        with open(url[7:], 'r') as file:
            content = file.read()
    else:
        response = requests.get(url)
        content = response.text

    soup = bs4.BeautifulSoup(content, 'html.parser')
    text = soup.get_text()

    phone_pattern = re.compile(r'\+\d{1,3}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}')
    phone_numbers = phone_pattern.findall(text)

    with open(output_path, 'w') as file:
        json.dump(phone_numbers, file)

    return phone_numbers