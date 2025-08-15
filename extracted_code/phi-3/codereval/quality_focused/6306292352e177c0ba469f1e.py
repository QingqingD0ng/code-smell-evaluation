import re

def process_text_links(text):
    # Convert textual HTML links to actual HTML
    text = re.sub(r'(?<!\w)([a-zA-Z][a-zA-Z0-9+.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'<a href="mailto:\1">\1</a>', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank">\1</a>', text)
    return text