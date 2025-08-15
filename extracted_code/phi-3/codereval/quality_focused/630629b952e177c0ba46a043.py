import requests
import os

def get_nodeinfo_well_known_document(url, document_path=None):
    response = requests.get(url)
    response.raise_for_status()
    
    if document_path:
        with open(document_path, 'w') as file:
            file.write(response.text)
    
    return {
        'url': url,
        'document_path': document_path,
        'content': response.text
    }