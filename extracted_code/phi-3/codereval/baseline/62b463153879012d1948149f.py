import os
import mimetypes

def _eval_file(prefix, file_path):
    if not os.path.isfile(file_path):
        return None
    
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if not mime_type:
        return None
    
    if prefix.lower() in mime_type.lower() or prefix.lower() == mime_type.lower():
        if mime_type.startswith('application/pdf'):
            return {'component_id': 'pdf_component', file_path: file_path, 'ftype': mime_type}
        else:
            return {'component_id': 'other_component', file_path: file_path, 'ftype': mime_type}
    
    return None