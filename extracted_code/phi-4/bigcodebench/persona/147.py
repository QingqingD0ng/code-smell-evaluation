import socket
from ipaddress import IPv4Network
from threading import Thread

def task_func(ip_range, port):
    def scan_ip(ip):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((str(ip), port)) == 0
                results[ip] = result
        except:
            results[ip] = False

    network = IPv4Network(ip_range)
    results = {}
    threads = []

    for ip in network.hosts():
        thread = Thread(target=scan_ip, args=(ip,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results