from xml.etree import ElementTree as ET
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

def verify_relayable_signature(public_key, doc, signature):
    # Parse the XML document
    root = ET.fromstring(doc)
    
    # Define the string to sign (assuming the document is an XML with a single root)
    data_to_sign = ET.tostring(root, encoding='utf-8', method='xml')
    
    # Load the public key
    public_key_obj = serialization.load_pem_public_key(
        public_key.encode('utf-8')
    )
    
    # Verify the signature
    try:
        public_key_obj.verify(
            signature,
            data_to_sign,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False