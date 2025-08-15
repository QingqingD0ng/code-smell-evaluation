import socket
from ipaddress import IPv4Network
from threading import Thread

def check_port(ip, port, results):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((ip, port)) == 0
            results[ip] = result
    except:
        results[ip] = False

def task_func(ip_range, port):
    network = IPv4Network(ip_range)
    results = {}
    threads = []

    for ip in network:
        ip_str = str(ip)
        thread = Thread(target=check_port, args=(ip_str, port, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results