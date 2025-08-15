import rsa
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
import os
import base64

def task_func():
    # Generate RSA key pair
    public_key, private_key = rsa.newkeys(512)

    # Generate random bytes for filename and encryption data
    random_bytes = get_random_bytes(8)
    filename = f"private_key_{random_bytes.hex()}.txt"
    password = get_random_bytes(16)
    nonce = get_random_bytes(16)

    # Encrypt the private key with AES
    cipher = AES.new(password, AES.MODE_EAX, nonce)
    encrypted_private_key = cipher.encrypt(private_key.save_pkcs1())

    # Save the encrypted private key to a file
    with open(filename, 'wb') as file:
        file.write(cipher.nonce)
        file.write(encrypted_private_key)

    return public_key, filename, password, nonce

pub_key, filename, password, nonce = task_func()
print(pub_key)
print(filename)
print(password)
print(nonce)