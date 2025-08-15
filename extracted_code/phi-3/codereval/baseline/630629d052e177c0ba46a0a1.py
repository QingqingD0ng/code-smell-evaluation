import xml.etree.ElementTree as ET
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

def verify_relayable_signature(public_key_pem, doc, signature):
    # Load the public key from PEM format
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode(),
        backend=serialization.default_backend()
    )

    # Parse the XML document
    root = ET.fromstring(doc)

    # Serialize the XML document for signing
    xml_to_sign = ET.tostring(root, encoding='utf-8', method='xml')

    # Verify the signature
    try:
        public_key.verify(
            signature,
            xml_to_sign,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False