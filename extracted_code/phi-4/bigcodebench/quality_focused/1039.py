import ssl
import os
import hashlib

def task_func(client_socket, cert_file, key_file, buffer_size=1024):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    with context.wrap_socket(client_socket, server_side=True) as secure_socket:
        try:
            file_path = secure_socket.recv(buffer_size).decode().strip()
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    file_hash = hashlib.sha256(file.read()).hexdigest()
                    secure_socket.sendall(file_hash.encode())
                    return file_hash
            else:
                error_message = 'File not found'
                secure_socket.sendall(error_message.encode())
                return error_message
        except Exception as e:
            error_message = f'Error: {str(e)}'
            secure_socket.sendall(error_message.encode())
            return error_message