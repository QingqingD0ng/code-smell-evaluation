import re

def unquote(name):
    return re.sub(r'^"|"$', '', name)