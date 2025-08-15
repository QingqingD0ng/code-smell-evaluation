import socket
from ipaddress import IPv4Network
from threading import Thread, Lock

def check_port(ip, port, results, lock):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect((ip, port))
            with lock:
                results[ip] = True
        except:
            with lock:
                results[ip] = False

def task_func(ip_range, port):
    network = IPv4Network(ip_range)
    results = {}
    lock = Lock()
    threads = []

    for ip in network.hosts():
        thread = Thread(target=check_port, args=(str(ip), port, results, lock))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return results