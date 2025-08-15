import socket
from ipaddress import IPv4Network
from threading import Thread

def task_func(ip_range, port):
    def scan_ip(ip, port, results):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect((str(ip), port))
            results[ip] = True
        except:
            results[ip] = False
        finally:
            sock.close()

    ip_network = IPv4Network(ip_range)
    threads = []
    results = {}

    for ip in ip_network.hosts():
        thread = Thread(target=scan_ip, args=(ip, port, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results