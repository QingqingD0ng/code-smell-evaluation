import rsa
import os
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES
from base64 import b64encode

def task_func():
    (pub_key, priv_key) = rsa.newkeys(2048)
    password = get_random_bytes(32)
    nonce = get_random_bytes(16)
    cipher = AES.new(password, AES.MODE_GCM, nonce=nonce)
    encrypted_priv_key, tag = cipher.encrypt_and_digest(priv_key.save_pkcs1())
    random_bytes = get_random_bytes(8)
    filename = f"private_key_{random_bytes.hex()}.txt"
    with open(filename, 'wb') as f:
        f.write(b64encode(nonce + tag + encrypted_priv_key))
    return pub_key, filename, password, nonce