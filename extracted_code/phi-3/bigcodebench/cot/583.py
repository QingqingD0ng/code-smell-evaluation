import rsa
import os
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from base64 import b64encode, b64decode

def task_func():
    file_name_prefix = os.urandom(8).hex()
    filename = f"private_key_{file_name_prefix}.txt"
    (pub_key, priv_key) = rsa.newkeys(512)
    password = get_random_bytes(16)
    nonce = get_random_bytes(16)
    cipher_aes = AES.new(password, AES.MODE_EAX, nonce)
    encrypted_priv_key = cipher_aes.encrypt(priv_key.save_pkcs1())
    ciphertext = b64encode(encrypted_priv_key)
    with open(filename, 'wb') as f:
        f.write(ciphertext)
    file_name = filename
    return pub_key, file_name, password, nonce

pub_key, file_name, password, nonce = task_func()