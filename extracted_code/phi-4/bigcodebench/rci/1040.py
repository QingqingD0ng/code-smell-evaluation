import socket
import select
import queue
from datetime import datetime, timedelta

BUFFER_SIZE = 1024
DEFAULT_RUN_DURATION = 5

def task_func(server_address="localhost", server_port=12345, buffer_size=BUFFER_SIZE, run_duration=DEFAULT_RUN_DURATION):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((server_address, server_port))
    server_socket.listen(5)
    server_socket.setblocking(False)

    inputs = [server_socket]
    outputs = []
    message_queues = {}

    end_time = datetime.now() + timedelta(seconds=run_duration)

    try:
        while inputs and datetime.now() < end_time:
            readable, writable, exceptional = select.select(inputs, outputs, inputs)

            for s in readable:
                if s is server_socket:
                    connection, client_address = s.accept()
                    connection.setblocking(False)
                    inputs.append(connection)
                    message_queues[connection] = queue.Queue()
                else:
                    data = s.recv(buffer_size)
                    if data:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        response = data + b''+ timestamp.encode()
                        message_queues[s].put(response)
                        if s not in outputs:
                            outputs.append(s)
                    else:
                        if s in outputs:
                            outputs.remove(s)
                        inputs.remove(s)
                        s.close()
                        del message_queues[s]

            for s in writable:
                try:
                    next_msg = message_queues[s].get_nowait()
                except queue.Empty:
                    outputs.remove(s)
                else:
                    s.send(next_msg)

            for s in exceptional:
                inputs.remove(s)
                if s in outputs:
                    outputs.remove(s)
                s.close()
                del message_queues[s]

    finally:
        for s in inputs:
            s.close()
        server_socket.close()

    return f"Server started on {server_address}:{server_port}. Ran for {run_duration} seconds."

# Example usage:
# print(task_func())