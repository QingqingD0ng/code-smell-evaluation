from cryptography.hazmat.primitives import serialization, hashes

def task_func(file_path):
    # Generate RSA keys
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # Encrypt the file with AES
    aes_key = os.urandom(32)  # AES-256 key
    iv = os.urandom(16)  # Initialization vector
    with open(file_path, 'rb') as f:
        data = f.read()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Encrypt AES key with RSA public key
    rsa_encryption = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Save files
    encrypted_file_path = file_path + '.encrypted'
    encrypted_key_path = file_path + '.encrypted_aes_key'
    with open(encrypted_file_path, 'wb') as f:
        f.write(encrypted_data)
    with open(encrypted_key_path, 'wb') as f:
        f.write(rsa_encryption)

    return public_key, encrypted_file_path, encrypted_key_path