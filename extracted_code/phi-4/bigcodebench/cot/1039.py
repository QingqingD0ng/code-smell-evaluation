import ssl
import os
import hashlib
import socket

def task_func(client_socket, cert_file, key_file, buffer_size=1024):
    try:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=cert_file, keyfile=key_file)

        secure_socket = context.wrap_socket(client_socket, server_side=True)

        file_path = secure_socket.recv(buffer_size).decode().strip()
        if not os.path.exists(file_path):
            return "File not found"

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(buffer_size):
                sha256_hash.update(chunk)
        
        file_hash = sha256_hash.hexdigest()
        secure_socket.sendall(file_hash.encode())
        return file_hash

    except Exception as e:
        return f"Error: {str(e)}"

    finally:
        secure_socket.close()