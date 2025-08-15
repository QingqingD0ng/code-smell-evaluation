import re

def process_text_links(text):
    # Define a regular expression pattern for URLs
    url_pattern = r'(https?://\S+)'
    
    # Function to add attributes to links and convert them into HTML anchor tags
    def add_attributes_and_linkify(match):
        url = match.group(0)
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>'

    # Replace textual links with HTML anchor tags
    text = re.sub(r'\b(https?://\S+)\b', add_attributes_and_linkify, text)
    
    # Add custom attributes to URLs
    def add_attributes(match):
        url = match.group(0)
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="custom-link" data-link-type="external" title="Click to visit the website">{url}</a>'

    text = re.sub(url_pattern, add_attributes, text)
    
    return text

# Example usage:
sample_text = "Check out this website: example.com or visit our docs at docs.example.com"
processed_text = process_text_links(sample_text)
print(processed_text)