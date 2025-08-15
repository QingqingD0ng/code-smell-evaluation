import requests
from urllib.parse import urlparse

def get_nodeinfo_well_known_document(url, document_path=None):
    parsed_url = urlparse(url)
    nodeinfo_url = f"{parsed_url.scheme}://{parsed_url.netloc}/nodeinfo"
    
    try:
        response = requests.get(nodeinfo_url)
        response.raise_for_status()
        node_info = response.json()
        result = {
            "url": nodeinfo_url,
            "document_path": node_info.get("documentPath") if document_path is None else document_path,
            "node_id": node_info.get("nodeId"),
            "name": node_info.get("name")
        }
    except requests.RequestException as e:
        result = {"error": str(e)}
    
    return result