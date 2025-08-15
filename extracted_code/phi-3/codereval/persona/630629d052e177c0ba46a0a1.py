import xml.etree.ElementTree as ET
from xmlsec import validate, xmlsec_constants
from xmlsec import XMLSecurityInfrastructure as xsi

def verify_relayable_signature(public_key, xml_string, signature):
    # Parse the XML string
    xml_root = ET.fromstring(xml_string)

    # Initialize the XML security infrastructure
    xsi_context = xsi.create_context()

    # Load the public key
    xsi_context.load_pubkey_x509(public_key)

    # Validate the XML against the DTD (if any) and the signature
    try:
        validate(xml_root, xml_string, xsi_context, xsi_context.load_x509_crypto_backend())
        print("Signature is valid.")
    except xmlsec.XMLSecurityInfrastructureError as e:
        if e.error_code == xmlsec.XMLSecurityInfrastructureError.CERTIFICATE_VERIFY_FAILED:
            print("Signature is invalid or the certificate cannot be verified.")
        else:
            print(f"An error occurred: {e}")
    finally:
        xsi_context.cleanup()