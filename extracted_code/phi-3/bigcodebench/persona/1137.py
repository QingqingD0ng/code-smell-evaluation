import bs4
import requests
import re
import json

def task_func(url: str, output_path: str) -> list:
    phone_numbers = []
    
    if url.startswith('http'):
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
    else:
        with open(url, 'r') as file:
            text = file.read()
    
    pattern = re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b')
    matches = pattern.findall(text)
    
    for number in matches:
        phone_numbers.append(number)
        
    with open(output_path, 'w') as file:
        json.dump(phone_numbers, file)
    
    return phone_numbers