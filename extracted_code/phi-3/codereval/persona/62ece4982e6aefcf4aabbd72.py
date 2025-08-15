import re

def unquote(name):
    return re.sub(r'^"(.*)"$', r'\1', name)