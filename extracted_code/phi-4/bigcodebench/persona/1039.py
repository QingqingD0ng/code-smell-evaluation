import ssl
import os
import hashlib

def task_func(client_socket, cert_file, key_file, buffer_size=1024):
    try:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        
        secure_socket = context.wrap_socket(client_socket, server_side=True)
        
        requested_file = secure_socket.recv(buffer_size).decode().strip()
        
        if not os.path.exists(requested_file):
            return 'File not found'
        
        sha256_hash = hashlib.sha256()
        
        with open(requested_file, "rb") as f:
            for byte_block in iter(lambda: f.read(buffer_size), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    except Exception as e:
        return f"Error: {str(e)}"