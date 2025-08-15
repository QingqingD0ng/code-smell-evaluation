import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as sym_padding

def encrypt_file(file_path, aes_key, public_key, output_aes_key_path, output_encrypted_path):
    # Initialize padding
    padder = sym_padding.PKCS7(algorithms.AES.block_size).padder()
    unpadder = sym_padding.PKCS7(algorithms.AES.block_size).unpadder()

    # Initialize AES cipher
    iv = os.urandom(16)
    cipher_aes = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor_aes = cipher_aes.encryptor()

    # Read input file and encrypt data
    with open(file_path, 'rb') as f:
        plaintext = f.read()

    # Encrypt data and pad it
    padded_data = padder.update(plaintext) + padder.finalize()
    # Encrypt the padded data
    encrypted_data = encryptor_aes.update(padded_data) + encryptor_aes.finalize()

    # Encrypt AES key with RSA public key
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Write the encrypted AES key to file
    with open(output_aes_key_path, 'wb') as f:
        f.write(encrypted_aes_key)

    # Write the encrypted data to a file