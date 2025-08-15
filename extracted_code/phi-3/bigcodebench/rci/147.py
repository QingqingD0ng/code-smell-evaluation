import socket
from ipaddress import IPv4Network
import asyncio

async def check_port(ip, host, port):
    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.close()
        await writer.wait_closed()
        return f'{host}:{port} open'
    except Exception as e:
        return f'{host}:{port} closed - {str(e)}'

async def scan_ip_range(ip_range, port):
    network = IPv4Network(ip_range)

    tasks = []
    for ip in network.hosts():
        host = str(ip)
        task = asyncio.create_task(check_port(host, host, port))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return dict(enumerate(results))

if __name__ == '__main__':
    ip_range = '192.168.0.0/24'
    port = 80
    results = asyncio.run(scan_ip_range(ip_range, port))
    print(results)