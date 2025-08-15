import bs4
import requests
import re
import json

def task_func(url: str, output_path: str) -> list:
    phone_numbers = []
    
    if url.startswith('file://'):
        with open(url[7:], 'r', encoding='utf-8') as file:
            content = file.read()
    else:
        response = requests.get(url)
        content = response.text
    
    soup = bs4.BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    
    pattern = r'\+?\d[\d\s\-]{8,}\d'
    phone_numbers = re.findall(pattern, text)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(phone_numbers, json_file)
    
    return phone_numbers