import rsa
import os
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64

def task_func(user_password):
    # Generate RSA key pair
    (public_key, private_key) = rsa.newkeys(512)
    
    # Generate random filename
    random_bytes = os.urandom(8)
    filename = f"private_key_{random_bytes.hex()}.txt"
    
    # Derive AES key and salt
    salt = get_random_bytes(16)
    aes_key = PBKDF2(user_password, salt, dkLen=32, count=1000000)
    
    # Encrypt the private key with AES
    cipher_aes = AES.new(aes_key, AES.MODE_EAX)
    private_key_unpadded = private_key.save_pkcs1()
    encrypted_private_key = cipher_aes.encrypt(pad(private_key_unpadded, AES.block_size))
    
    # Save the encrypted private key to a file
    with open(filename, 'wb') as f:
        f.write(encrypted_private_key)
        f.write(cipher_aes.nonce)
    
    return public_key, filename, salt

# Example usage
user_password = input("Enter a password to encrypt your private key: ").encode()
pub_key, filename, salt = task_func(user_password)
assert isinstance(pub_key, rsa.PublicKey)
assert isinstance(filename, str)