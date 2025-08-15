import ssl
import socket

def task_func(client_socket, cert_file, key_file, buffer_size=1024):
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    secure_socket = context.wrap_socket(client_socket, server_side=True)

    try:
        file_path = secure_socket.recv(buffer_size).decode()
        if not os.path.isfile(file_path):
            return 'File not found'

        with open(file_path, 'rb') as file_to_hash:
            file_content = file_to_hash.read()
            file_hash = hashlib.sha256(file_content).hexdigest()

        return file_hash
    except Exception as e:
        return f"An error occurred: {str(e)}"