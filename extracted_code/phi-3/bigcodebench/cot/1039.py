import ssl
import os
import hashlib
from socket import socket, AF_INET, SOCK_STREAM

def task_func(client_socket, cert_file, key_file, buffer_size=1024):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    
    secure_socket = context.wrap_socket(client_socket, server_side=True)
    
    try:
        file_path = secure_socket.recv(buffer_size).decode()
        if not os.path.exists(file_path):
            return 'File not found'
        
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as file:
            while True:
                data = file.read(buffer_size)
                if not data:
                    break
                sha256_hash.update(data)
        return sha256_hash.hexdigest()
    except Exception as e:
        return f'Error: {str(e)}'