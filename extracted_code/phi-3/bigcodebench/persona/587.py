import rsa

import os

from cryptography.hazmat.backends import default_backend

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from cryptography.hazmat.primitives import padding

from base64 import b64encode


def task_func(file_path):
    # Generate RSA keys
    (public_key, private_key) = rsa.newkeys(2048)

    # Read the original file
    with open(file_path, 'rb') as file:
        original_data = file.read()

    # Pad the data to be AES block size compatible
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(original_data) + padder.finalize()

    # Encrypt the data with AES
    aes_key = os.urandom(32)  # AES-256 key
    iv = os.urandom(16)  # AES IV
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Encrypt the AES key with RSA public key
    encrypted_aes_key = public_key.encrypt(aes_key, rsa.constants.OAEP_SIGNATURE + rsa.constants.PSS_SIGNATURE)

    # Save the encrypted file and encrypted AES key
    encrypted_file_path = file_path + '.encrypted'
    encrypted_key_file_path = file_path + '.encrypted_key'
    with open(encrypted_file_path, 'wb') as enc_file, open(encrypted_key_file_path, 'wb') as key_file:
        key_file.write(b64encode(encrypted_aes_key))
        enc_file.write(encrypted_data)