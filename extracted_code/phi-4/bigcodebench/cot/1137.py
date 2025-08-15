import requests
from bs4 import BeautifulSoup
import re
import json
import os

def task_func(url: str, output_path: str) -> list:
    phone_pattern = re.compile(r'\+?\d[\d\s\-().]{7,}\d')

    if url.startswith('file://'):
        file_path = url[7:]
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    else:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.get_text()

    phone_numbers = phone_pattern.findall(content)

    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(phone_numbers, json_file, ensure_ascii=False, indent=4)

    return phone_numbers