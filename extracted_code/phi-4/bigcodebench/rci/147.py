import socket
from ipaddress import IPv4Network
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

def task_func(ip_range, port):
    def check_port_open(ip, port, results, lock):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                sock.connect((str(ip), port))
                with lock:
                    results[ip] = True
            except (socket.timeout, ConnectionRefusedError, OSError):
                with lock:
                    results[ip] = False

    ip_network = IPv4Network(ip_range)
    results = {}
    lock = Lock()

    with ThreadPoolExecutor(max_workers=100) as executor:
        for ip in ip_network.hosts():
            executor.submit(check_port_open, ip, port, results, lock)

    return results