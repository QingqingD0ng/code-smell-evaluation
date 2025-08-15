import ssl
import os
import hashlib
import socket
from typing import Optional

BUFFER_SIZE = 1024
FILE_NOT_FOUND_MESSAGE = 'File not found'
ERROR_MESSAGE_PREFIX = 'Error: '

def create_ssl_context(cert_file: str, key_file: str) -> ssl.SSLContext:
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return context

def task_func(client_socket: socket.socket, cert_file: str, key_file: str, buffer_size: int = BUFFER_SIZE) -> Optional[str]:
    context = create_ssl_context(cert_file, key_file)
    ssl_socket = context.wrap_socket(client_socket, server_side=True)
    try:
        data = ssl_socket.recv(buffer_size).decode('utf-8').strip()
        if not os.path.isabs(data) or '..' in data:
            raise ValueError("Invalid file path")
        
        if os.path.exists(data) and os.path.isfile(data):
            sha256_hash = hashlib.sha256()
            with open(data, 'rb') as file:
                for byte_block in iter(lambda: file.read(buffer_size), b""):
                    sha256_hash.update(byte_block)
            file_hash = sha256_hash.hexdigest()
            ssl_socket.sendall(file_hash.encode('utf-8'))
        else:
            ssl_socket.sendall(FILE_NOT_FOUND_MESSAGE.encode('utf-8'))
    except (OSError, ValueError) as e:
        ssl_socket.sendall((ERROR_MESSAGE_PREFIX + str(e)).encode('utf-8'))
    except Exception as e:
        ssl_socket.sendall((ERROR_MESSAGE_PREFIX + "Unexpected error").encode('utf-8'))
    finally:
        ssl_socket.close()