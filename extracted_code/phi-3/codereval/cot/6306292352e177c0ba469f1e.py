import re

def process_text_links(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    
    def add_email_attributes(match):
        return f'<a href="mailto:{match.group()}" rel="noopener noreferrer" class="email-link">{match.group()}</a>'
    
    def add_url_attributes(match):
        return f'<a href="{match.group()}" target="_blank" rel="noopener noreferrer" class="url-link">{match.group()}</a>'
    
    text = re.sub(email_pattern, add_email_attributes, text)
    text = re.sub(url_pattern, add_url_attributes, text)
    
    return text