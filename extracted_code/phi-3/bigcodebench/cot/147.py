import socket
import ipaddress
from threading import Thread

def is_port_open(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((ip, port)) == 0

def scan_ip_range(ip_range, port):
    ip_network = ipaddress.IPv4Network(ip_range)
    results = {}
    threads = []

    def worker(ip):
        results[str(ip)] = is_port_open(str(ip), port)

    for ip in ip_network.hosts():
        thread = Thread(target=worker, args=(ip,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results

if __name__ == "__main__":
    scan_results = scan_ip_range('192.168.0.0/24', 80)
    print(scan_results)