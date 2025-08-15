import socket
import select
import queue
from datetime import datetime, timedelta

def task_func(
    server_address="localhost", 
    server_port=12345, 
    buffer_size=1024, 
    run_duration=5
):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setblocking(0)
    server_socket.bind((server_address, server_port))
    server_socket.listen(5)

    inputs = [server_socket]
    outputs = []
    message_queue = queue.Queue()

    start_time = datetime.now()

    while datetime.now() - start_time < timedelta(seconds=run_duration):
        readable, writable, exceptional = select.select(inputs, outputs, inputs)

        for s in readable:
            if s is server_socket:
                client_socket, client_address = s.accept()
                client_socket.setblocking(0)
                inputs.append(client_socket)
            else:
                data = s.recv(buffer_size)
                if data:
                    message = f"{datetime.now().isoformat()} {data.decode()}"
                    outputs.append(s)
                    message_queue.put(message)

        for s in writable:
            if not message_queue.empty():
                message = message_queue.get()
                s.send(message.encode())

        for s in exceptional:
            inputs.remove(s)
            if s in outputs:
                outputs.remove(s)
            s.close()

    server_socket.close()
    return f"Server started on {server_address}:{server_port}. Ran for {run_duration} seconds."

# Example usage
print(task_func())