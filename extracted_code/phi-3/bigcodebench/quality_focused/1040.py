import socket
import select
import queue
from datetime import datetime, timedelta

def task_func(server_address="localhost", server_port=12345, buffer_size=1024, run_duration=5):
    sockets_list = []
    message_queue = queue.Queue()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setblocking(0)
    server_socket.bind((server_address, server_port))
    server_socket.listen(5)
    sockets_list.append(server_socket)

    while True:
        read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list, 0.1)

        for sock in read_sockets:
            if sock == server_socket:
                client_socket, client_address = server_socket.accept()
                sockets_list.append(client_socket)
                print(f'Client connected from {client_address}')
            else:
                data = sock.recv(buffer_size)
                if not data:
                    sock.close()
                    sockets_list.remove(sock)
                    continue
                message_queue.put(data)
                response = f"Server time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".encode()
                for sock in sockets_list:
                    if sock!= server_socket and sock!= sock.accept():
                        sock.send(response)

        for sock in exception_sockets:
            sockets_list.remove(sock)
            sock.close()

        if len(sockets_list) == 1 and sock == server_socket:
            break

        if datetime.now() - start_time >= timedelta(seconds=run_duration):
            break

    return f"Server started on {server_address}:{server_port}. Ran for {run_duration} seconds."

start_time = datetime.now()
print(task_