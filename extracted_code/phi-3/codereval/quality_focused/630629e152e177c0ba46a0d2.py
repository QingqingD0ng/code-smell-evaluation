import requests
from xml.etree import ElementTree

def retrieve_and_parse_diaspora_webfinger(handle):
    url = f"https://diaspora.us/@{handle}/webfinger"
    response = requests.get(url)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    xml_data = response.text

    root = ElementTree.fromstring(xml_data)
    webfinger_dict = {
        'url': root.find('link').attrib['href'],
        'fullname': root.find('preferred').text,
        'displayname': root.find('displayname').text
    }

    return webfinger_dict