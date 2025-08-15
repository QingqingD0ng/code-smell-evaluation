import socket
from ipaddress import IPv4Network
from threading import Thread

def check_port(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((ip, port)) == 0

def task_func(ip_range, port):
    network = IPv4Network(ip_range)
    results = {}
    threads = []

    for ip in network.hosts():
        ip_str = str(ip)
        thread = Thread(target=lambda p=port, r=results: results.update({p: check_port(r, p)}))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results