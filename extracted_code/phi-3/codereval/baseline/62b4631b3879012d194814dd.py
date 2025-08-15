import re

def fix_namespace_prefix_w(content):
    return re.sub(r'\bw:st=', 'w-st=', content)