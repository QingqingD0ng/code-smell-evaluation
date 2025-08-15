import re

def is_valid_octet(octet):
    octet_regex = r'^(25[0-5]|2[0-4]\d|[01]?\d\d?)$'
    return bool(re.match(octet_regex, octet))

def is_ipv4(target):
    octets = target.split('.')
    if len(octets)!= 4:
        return False

    for octet in octets:
        if not is_valid_octet(octet):
            return False
    
    return True