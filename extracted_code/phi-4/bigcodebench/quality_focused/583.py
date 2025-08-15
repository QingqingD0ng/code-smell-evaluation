import rsa
import os
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from base64 import b64encode, b64decode

def task_func():
    # Generate RSA key pair
    (pub_key, priv_key) = rsa.newkeys(2048)

    # Generate a random password and nonce for AES encryption
    password = get_random_bytes(16)
    nonce = get_random_bytes(16)

    # Encrypt the private key using AES
    cipher = AES.new(password, AES.MODE_EAX, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(priv_key.save_pkcs1())

    # Create a filename using random bytes
    random_bytes = get_random_bytes(8)
    filename = f"private_key_{random_bytes.hex()}.txt"

    # Write the encrypted private key to the file
    with open(filename, 'wb') as file:
        file.write(b64encode(ciphertext))
        file.write(b64encode(tag))
        file.write(b64encode(nonce))

    return pub_key, filename, password, nonce