import rsa
import os
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from base64 import b64encode, b64decode

def task_func():
    # Generate RSA key pair
    (pub_key, priv_key) = rsa.newkeys(2048)
    
    # Generate random bytes for filename and password
    random_bytes = get_random_bytes(8)
    password = get_random_bytes(32)
    nonce = get_random_bytes(16)
    
    # Format the filename with hex representation of random bytes
    filename = f"private_key_{random_bytes.hex()}.txt"
    
    # Encrypt the private key with AES
    cipher = AES.new(password, AES.MODE_EAX, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(priv_key.save_pkcs1('PEM'))
    
    # Save encrypted private key to file
    with open(filename, 'wb') as f:
        f.write(nonce + tag + ciphertext)
    
    # Return public key, filename, password, and nonce
    return pub_key, filename, password, nonce