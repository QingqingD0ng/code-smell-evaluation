import bs4
import requests
import re
import json

def task_func(url: str, output_path: str) -> list:
    if url.startswith('file://'):
        with open(url[7:], 'r') as file:
            content = file.read()
    else:
        response = requests.get(url)
        content = response.text

    soup = bs4.BeautifulSoup(content, 'html.parser')
    phone_numbers = re.findall(r'\+?[1-9]\d{1,14}', content)

    with open(output_path, 'w') as file:
        json.dump(phone_numbers, file)

    return phone_numbers


# Example usage:
# task_func('https://example.com', 'output.json')
# task_func('file:///path/to/local/file.txt', 'output.json')