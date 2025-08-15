import bs4
import requests
import re
import json

def task_func(url: str, output_path: str) -> list:
    phone_numbers = []

    # Regular expression pattern to match phone numbers
    phone_pattern = re.compile(r'\+?\d[\d\s()-]{8,}\d')

    if url.startswith('file://'):
        # Handle local file
        file_path = url[7:]  # Strip 'file://' prefix
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    else:
        # Handle URL
        response = requests.get(url)
        content = response.text

    # Use BeautifulSoup to parse the content
    soup = bs4.BeautifulSoup(content, 'html.parser')

    # Extract text from the webpage
    text = soup.get_text()

    # Find all phone numbers using the regex pattern
    matches = phone_pattern.findall(text)
    phone_numbers = [match.strip() for match in matches]

    # Save the phone numbers to a JSON file
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(phone_numbers, json_file, ensure_ascii=False, indent=4)

    return phone_numbers