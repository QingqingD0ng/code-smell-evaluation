import rsa
import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from base64 import b64encode, b64decode

def task_func():
    # Generate RSA key pair
    (pub_key, priv_key) = rsa.newkeys(2048)

    # Generate a random password for AES encryption
    password = get_random_bytes(16)

    # Generate a random nonce for AES encryption
    nonce = get_random_bytes(16)

    # Encrypt the private key using AES
    cipher = AES.new(password, AES.MODE_EAX, nonce=nonce)
    encrypted_priv_key, tag = cipher.encrypt_and_digest(priv_key.save_pkcs1())

    # Generate a random 8-byte sequence and convert to hex for the filename
    random_bytes = get_random_bytes(8)
    filename = f"private_key_{random_bytes.hex()}.txt"

    # Save the encrypted private key to a file
    with open(filename, 'wb') as f:
        f.write(b64encode(encrypted_priv_key))

    return pub_key, filename, password, nonce