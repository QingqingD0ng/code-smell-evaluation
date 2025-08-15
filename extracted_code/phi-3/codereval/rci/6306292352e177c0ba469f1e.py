import re
from html.parser import HTMLParser

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(self._create_link_dict(href))

    def _create_link_dict(self, href):
        return {'text': f'<a href="{href}" target="_blank" rel="noopener noreferrer">{href}</a>', 'href': href}

def extract_links(text):
    parser = LinkParser()
    parser.feed(text)
    return parser.links

def add_attributes_to_links(links):
    return [{'text': link['text'].split('>')[0], 'href': link['href'], 'target': '_blank','rel': 'noopener noreferrer'} for link in links]

def linkify_textual_links(text):
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    def replace_link(match):
        link_text = match.group(0)
        return f'<a href="{match.group(2)}" target="_blank" rel="noopener noreferrer">{link_text}</a>'
    return re.sub(pattern, replace_link, text)

def process_text_links(text):
    links = extract_links(text)
    links_with_attributes = add_attributes_to_links(links)
    text_with_linkified_links = linkify_textual_links(text)
    return text_with_linkified_links, links_with_attributes