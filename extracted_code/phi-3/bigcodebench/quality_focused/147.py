import socket
import ipaddress
from threading import Thread

def scan_port(ip, port, results):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((ip, port))
            results[ip] = result == 0
    except Exception:
        results[ip] = False

def task_func(ip_range, port):
    net = ipaddress.IPv4Network(ip_range)
    threads = []
    results = {}
    for ip in net:
        thread = Thread(target=scan_port, args=(str(ip), port, results))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    return results