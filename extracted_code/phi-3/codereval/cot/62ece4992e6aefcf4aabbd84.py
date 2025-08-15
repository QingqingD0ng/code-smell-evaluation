import re

def is_ipv4(target):
    pattern = r'^(\d{1,2}|1\d{2}|2[0-4]\d|25[0-5])\.(\d{1,2}|1\d{2}|2[0-4]\d|25[0-5])\.(\d{1,2}|1\d{2}|2[0-4]\d|25[0-5])\.(\d{1,2}|1\d{2}|2[0-4]\d|25[0-5])$'
    return bool(re.match(pattern, target))

# Example usage:
print(is_ipv4("192.168.1.1"))  # Should return True
print(is_ipv4("256.1.1.1"))    # Should return False