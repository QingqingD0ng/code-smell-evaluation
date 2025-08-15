import socket
import select
import queue
from datetime import datetime, timedelta

def task_func(server_address="localhost", server_port=12345, buffer_size=1024, run_duration=5):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setblocking(0)
    server_socket.bind((server_address, server_port))
    server_socket.listen(5)
    server_socket.setblocking(0)

    read_sockets = [server_socket]
    exception_sockets = []

    message_queue = queue.Queue()

    start_time = datetime.now()

    try:
        while datetime.now() - start_time < timedelta(seconds=run_duration):
            readable, writable, exceptional = select.select(
                read_sockets, [], exception_sockets, max(0, run_duration - (datetime.now() - start_time).total_seconds())
            )

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

            for sock in exceptional:
                sock.close()
                read_sockets.remove(sock)

    except Exception as e:
        print(f"Exception occurred: {str(e)}")

    return "Server finished running."