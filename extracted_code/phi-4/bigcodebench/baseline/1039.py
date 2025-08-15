import ssl
import os
import hashlib
import socket

def task_func(client_socket, cert_file, key_file, buffer_size=1024):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    ssl_socket = context.wrap_socket(client_socket, server_side=True)

    try:
        data = ssl_socket.recv(buffer_size).decode('utf-8').strip()
        if os.path.exists(data):
            sha256_hash = hashlib.sha256()
            with open(data, 'rb') as file:
                for byte_block in iter(lambda: file.read(buffer_size), b""):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
            ssl_socket.sendall(file_hash.encode('utf-8'))
        else:
            ssl_socket.sendall('File not found'.encode('utf-8'))
    except Exception as e:
        ssl_socket.sendall(f'Error: {str(e)}'.encode('utf-8'))
    finally:
        ssl_socket.close()