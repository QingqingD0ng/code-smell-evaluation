import xml.etree.ElementTree as ET
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

def load_public_key(pem_str):
    return serialization.load_pem_public_key(
        pem_str.encode(),
        backend=default_backend()
    )

def serialize_xml(root):
    return ET.tostring(root, encoding='utf-8', method='xml')

def verify_signature(public_key, xml_data, signature):
    try:
        public_key.verify(
            signature,
            xml_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False

def verify_relayable_signature(public_key_pem, doc, signature):
    root = ET.fromstring(doc)
    xml_data = serialize_xml(root)
    public_key = load_public_key(public_key_pem)
    return verify_signature(public_key, xml_data, signature)