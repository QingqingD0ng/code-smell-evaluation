import xml.etree.ElementTree as ET
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

def verify_relayable_signature(public_key_pem, signed_xml, signature):
    root = ET.fromstring(signed_xml)
    root_element = root.find('.//Signature')
    if root_element is None:
        raise ValueError("Signature not found in the XML.")
    signature_value = root_element.text.encode()
    public_key = serialization.load_pem_public_key(
        public_key_pem.encode()
    )
    try:
        public_key.verify(
            signature_value,
            signed_xml.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        print("Signature is valid.")
    except InvalidSignature:
        print("Signature is invalid.")