import bs4
import requests
import re
import json
from urllib.parse import urlparse

def task_func(url: str, output_path: str) -> list:
    def extract_phone_numbers(text: str) -> list:
        pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')
        return pattern.findall(text)

    def get_content(url: str) -> str:
        if url.startswith("file://"):
            with open(urlparse(url).path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            response = requests.get(url)
            response.raise_for_status()
            return response.text

    def save_to_json(data: list, path: str):
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    content = get_content(url)
    soup = bs4.BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    phone_numbers = extract_phone_numbers(text)
    formatted_numbers = [''.join(number) for number in phone_numbers]
    save_to_json(formatted_numbers, output_path)
    return formatted_numbers