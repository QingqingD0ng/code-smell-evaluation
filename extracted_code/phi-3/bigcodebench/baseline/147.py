import socket
from ipaddress import IPv4Network
from threading import Thread

def task_func(ip_range, port):
    open_ports = {}

    def check_port(ip):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((str(ip), port))
            open_ports[str(ip)] = result == 0

    network = IPv4Network(ip_range)
    threads = []

    for ip in network.hosts():
        t = Thread(target=check_port, args=(ip,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return open_ports