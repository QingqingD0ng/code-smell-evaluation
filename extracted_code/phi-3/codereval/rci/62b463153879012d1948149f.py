import os

import mimetypes


PDF_MIME_TYPE = 'application/pdf'

COMPONENT_ID_PDF = 'pdf_component'

COMPONENT_ID_OTHER = 'other_component'


def _eval_file(prefix, file_path):
    if not os.path.isfile(file_path):
        return None
    
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if not mime_type:
        return None
    
    if prefix.lower() in mime_type.lower() or prefix.lower() == mime_type.lower():
        return {COMPONENT_ID_PDF: file_path, 'ftype': mime_type} if mime_type.startswith(PDF_MIME_TYPE) else {COMPONENT_ID_OTHER: file_path, 'ftype': mime_type}
    
    return None