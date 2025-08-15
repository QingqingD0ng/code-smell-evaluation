import socket
from ipaddress import ip_address

def is_local(host):
    # Check if the host is already a local IP address
    if ip_address(host) in ip_address(socket.gethostbyname('localhost')):
        return True

    # Check if the hostname is a valid local hostname
    try:
        hostname = socket.gethostbyaddr(host)[0]
        if hostname == 'localhost' or hostname == '127.0.0.1':
            return True
    except socket.error:
        pass

    return False