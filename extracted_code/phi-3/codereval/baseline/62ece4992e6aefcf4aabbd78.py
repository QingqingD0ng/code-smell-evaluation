import socket

def is_local(host):
    local_ips = ['127.0.0.1', 'localhost']
    for local_ip in local_ips:
        if host == local_ip:
            return True
    try:
        socket.gethostbyname(host)
        return True
    except socket.error:
        return False