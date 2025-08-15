import rsa
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
import base64

def task_func(file_path):
    # Generate RSA keys
    (pub_key, priv_key) = rsa.newkeys(512)

    # Read the file content
    with open(file_path, 'rb') as f:
        file_data = f.read()

    # Encrypt the file with AES
    aes_key = os.urandom(32)
    iv = os.urandom(16)
    aes_cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = aes_cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(file_data) + padder.finalize()
    aes_encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Encrypt the AES key with RSA public key
    encrypted_aes_key = pub_key.encrypt(aes_key, None)

    # Save the encrypted file
    encrypted_file_name = f"{file_path}.encrypted"
    with open(encrypted_file_name, 'wb') as f:
        f.write(aes_encrypted_data)

    # Save the encrypted AES key
    encrypted_key_name = f"{file_path}.encrypted_key"
    with open(encrypted_key_name, 'wb') as f:
        f.write(base64.b64encode(encrypted_aes_key))

    return pub_key, encrypted_file_name, encrypted_key_name

# Example usage:
# pub_key, encrypted_file, encrypted_key_file = task_func('my_file.txt')
# print(