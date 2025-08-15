import bs4
import requests
import re
import json
from urllib.parse import urlparse

def fetch_content(url: str) -> str:
    if url.startswith('file://'):
        file_path = urlparse(url).path
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise RuntimeError(f"Error reading file: {file_path}") from e
    else:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise RuntimeError("Error fetching URL") from e

def extract_phone_numbers(text: str) -> list:
    pattern = r'\+?\d[\d\s\-]{8,}\d'
    return re.findall(pattern, text)

def save_to_json(data: list, output_path: str):
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file)
    except Exception as e:
        raise RuntimeError(f"Error writing to JSON file: {output_path}") from e

def task_func(url: str, output_path: str) -> list:
    content = fetch_content(url)
    soup = bs4.BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    phone_numbers = extract_phone_numbers(text)
    save_to_json(phone_numbers, output_path)
    return phone_numbers