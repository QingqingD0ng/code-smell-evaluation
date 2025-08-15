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
                self.links.append({'text': self.get_starttag_text(), 'href': href})

def add_attributes_to_links(links):
    for link in links:
        link['target'] = '_blank'
        link['rel'] = 'noopener noreferrer'
    return links

def linkify_textual_links(text):
    def replace_link(match):
        link_text = match.group(0)
        return f'<a href="{match.group(1)}" target="_blank" rel="noopener noreferrer">{link_text}</a>'

    pattern = r'(\[([^\]]+)\]\(([^)]+)\))'
    return re.sub(pattern, replace_link, text)

def process_text_links(text):
    link_parser = LinkParser()
    link_parser.feed(text)
    links_with_attributes = add_attributes_to_links(link_parser.links)
    text_with_linkified_links = linkify_textual_links(text)
    return text_with_linkified_links, links_with_attributes