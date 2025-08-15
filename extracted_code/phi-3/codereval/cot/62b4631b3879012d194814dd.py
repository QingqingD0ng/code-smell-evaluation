import re

def fix_namespace_prefix_w(content):
    return re.sub(r'\bw:\w+=', r'w-\g<1>', content)