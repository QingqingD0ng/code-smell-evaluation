import re

def is_ipv4(target):
    pattern = r'^(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.' \
             r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.' \
             r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.' \
             r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$'
    return bool(re.match(pattern, target))