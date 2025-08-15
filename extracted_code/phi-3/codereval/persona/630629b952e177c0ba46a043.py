import requests

def get_nodeinfo_well_known_document(url, document_path=None):
    response = requests.get(url)
    response.raise_for_status()
    document_content = response.text

    node_info = {
        'url': url,
        'document_path': document_path,
        'content': document_content
    }

    return node_info