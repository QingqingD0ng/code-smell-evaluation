import socket
import ipaddress

def is_local(host):
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback
    except ValueError:
        try:
            resolved_ip = socket.gethostbyname(host)
            return ipaddress.ip_address(resolved_ip).is_loopback
        except socket.gaierror:
            return False