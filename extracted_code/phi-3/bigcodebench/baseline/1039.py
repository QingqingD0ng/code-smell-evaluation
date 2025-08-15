import ssl
import os
import hashlib
import socket

def task_func(client_socket, cert_file, key_file, buffer_size=1024):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    
    secure_socket = context.wrap_socket(client_socket, server_side=True)
    
    try:
        file_path = secure_socket.recv(buffer_size).decode()
        if not os.path.isfile(file_path):
            return 'File not found'
        
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(buffer_size), b''):
                sha256_hash.update(chunk)
                
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage:
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind(('localhost', 443))
# server_socket.listen(5)
# cert_file = "path/to/certificate.crt"
# key_file = "path/to/private.key"
# client_socket, addr = server_socket.accept()
# file_hash = task_func(client_socket, cert_file, key_file)
# print("Sent file hash:", file_hash)
# server_socket.close()