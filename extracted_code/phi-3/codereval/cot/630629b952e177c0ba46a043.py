import requests

def get_nodeinfo_well_known_document(url, document_path=None):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        node_info = {
            "url": url,
            "document_path": document_path if document_path else data.get("document_path", ""),
            "data": data
        }
        return node_info
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
# node_info = get_nodeinfo_well_known_document("https://example.com/well-known-document")
# print(node_info)