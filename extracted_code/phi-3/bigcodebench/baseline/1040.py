import socket
import select
import queue
from datetime import datetime, timedelta

def task_func(server_address="localhost", server_port=12345, buffer_size=1024, run_duration=5):
    # Set up the server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setblocking(0)
    server_socket.bind((server_address, server_port))
    server_socket.listen(5)
    server_socket.setblocking(0)

    # Lists for read sockets and exceptions
    read_sockets = [server_socket]
    exception_sockets = []

    # Queue for storing received messages
    message_queue = queue.Queue()

    # Time when the server starts running
    start_time = datetime.now()

    try:
        while True:
            # Use select to wait for socket events
            readable, writable, exceptional = select.select(
                read_sockets, [], exception_sockets, run_duration - (datetime.now() - start_time).total_seconds()
            )

            if readable:
                for sock in readable:
                    if sock == server_socket:
                        client_socket, client_address = server_socket.accept()
                        client_socket.setblocking(0)
                        read_sockets.append(client_socket)
                    else:
                        try:
                            data = sock.recv(buffer_size)
                            if data:
                                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                data += current_time.encode()
                                message_queue.put(data)
                            else:
                                sock.close()
                                read_sockets.remove(sock)
                        except socket.error as e:
                            exceptional.append(sock)

            if exceptional:
                for sock in exceptional:
                    sock.close()
                    read_sockets.remove(sock)