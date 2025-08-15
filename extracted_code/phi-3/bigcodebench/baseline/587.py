from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes

def task_func(file_path):
    # Generate RSA keys
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # Load the original file
    with open(file_path, 'rb') as f:
        data = f.read()

    # Encrypt the file content with AES
    aes_key = os.urandom(32)  # AES-256 key
    iv = os.urandom(16)  # AES block size
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    cipher_aes = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor_aes = cipher_aes.encryptor()
    encrypted_data = encryptor_aes.update(padded_data) + encryptor_aes.finalize()

    # Encrypt the AES key with the RSA public key
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Save the encrypted AES key and the encrypted file
    encrypted_key_file = os.path.splitext(file_path)[0] + '.encrypted_aes_key'
    encrypted_file = os.path.splitext(file_path)[0] + '.encrypted'
    with open(encrypted_key_file, 'wb