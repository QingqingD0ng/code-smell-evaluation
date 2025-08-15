import os
import rsa
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad
from base64 import b64encode, b64decode

def generate_keypair():
    (public_key, private_key) = rsa.newkeys(512)
    return public_key, private_key

def encrypt_private_key(private_key, password):
    salt = get_random_bytes(16)
    private_key_encrypted = AES.new(PBKDF2(password, salt, dkLen=32), AES.MODE_EAX)
    private_key_encrypted_pem = private_key_encrypted.encrypt(private_key.save_pkcs1().encode('utf-8'))
    nonce = private_key_encrypted.nonce
    return private_key_encrypted_pem, salt, nonce

def save_key_to_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

def task_func():
    password = get_random_bytes(32)
    public_key, private_key = generate_keypair()
    encrypted_private_key_pem, salt, nonce = encrypt_private_key(private_key, password)
    filename = "private_key_" + b64encode(os.urandom(8)).decode('utf-8') + ".txt"
    save_key_to_file(filename, encrypted_private_key_pem)
    return public_key, filename, password, salt, nonce

pub_key, filename, password, salt, nonce = task_func()