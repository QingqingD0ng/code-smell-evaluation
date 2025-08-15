import bs4
import requests
import re
import json
from typing import List

def read_content_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

def extract_phone_numbers(content: str) -> List[str]:
    phone_numbers = re.findall(r'\+?[1-9]\d{1,14}', content)
    return phone_numbers

def write_phone_numbers_to_file(phone_numbers: List[str], output_path: str) -> None:
    with open(output_path, 'w') as file:
        json.dump(phone_numbers, file)

def task_func(url: str, output_path: str) -> List[str]:
    if not url.startswith('http'):
        url = 'file://' + url

    content = read_content_from_url(url)
    phone_numbers = extract_phone_numbers(content)
    write_phone_numbers_to_file(phone_numbers, output_path)

    return phone_numbers


# Example usage:
# task_func('https://example.com', 'output.json')
# task_func('file:///path/to/local/file.txt', 'output.json')