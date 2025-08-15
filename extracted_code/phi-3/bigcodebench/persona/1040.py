import socket
import select
import queue
from datetime import datetime, timedelta

def task_func(
    server_address="localhost", server_port=12345, buffer_size=1024, run_duration=5
):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setblocking(0)
    server_socket.bind((server_address, server_port))
    server_socket.listen()
    server_sockets = [server_socket]
    client_sockets = []
    received_data_queue = queue.Queue()
    server_time = datetime.now()

    try:
        while True:
            readable, _, _ = select.select(server_sockets + client_sockets, [], [])
            for sock in readable:
                if sock in server_sockets:
                    client_socket, addr = server_socket.accept()
                    client_socket.setblocking(0)
                    client_sockets.append(client_socket)
                else:
                    data = sock.recv(buffer_size)
                    if data:
                        server_time = datetime.now()
                        data += f" at {server_time.strftime('%Y-%m-%d %H:%M:%S')}".encode()
                        received_data_queue.put(data)
                        sock.sendall(data)
                    else:
                        client_sockets.remove(sock)
                        sock.close()
            if not client_sockets:
                break
            if datetime.now() - server_time >= timedelta(seconds=run_duration):
                break
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        server_socket.close()
        client_sockets = []
        received_data_queue.close()
        return f"Server started on {server_address}:{server_port}. Ran for {run_duration} seconds."