import socket

def is_local(host):
    local_hosts = ['localhost', '127.0.0.1', '::1']
    try:
        socket.inet_aton(host)
    except socket.error:
        host_ip = socket.gethostbyname(host)
    else:
        host_ip = host

    return host_ip in local_hosts