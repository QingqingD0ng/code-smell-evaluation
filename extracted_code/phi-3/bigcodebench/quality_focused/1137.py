import bs4
import requests
import re
import json

def task_func(url: str, output_path: str) -> list:
    phone_numbers = []

    if url.startswith('http://') or url.startswith('https://'):
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text
    else:
        # Read the local file
        with open(url, 'r') as file:
            html_content = file.read()

    # Parse the HTML content
    soup = bs4.BeautifulSoup(html_content, 'html.parser')

    # Extract phone numbers using regex
    for element in soup.find_all(text=re.compile(r'\+?[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{4,9}')):
        phone_numbers.append(element.strip())

    # Save the phone numbers in JSON format
    with open(output_path, 'w') as json_file:
        json.dump(phone_numbers, json_file, indent=4)

    return phone_numbers