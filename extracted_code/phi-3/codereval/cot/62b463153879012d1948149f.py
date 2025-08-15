import os
import mimetypes

def _eval_file(prefix, file_path):
    if not file_path.startswith(prefix):
        return None
    
    file_type, _ = mimetypes.guess_type(file_path)
    if file_type and file_type.startswith('application/pdf'):
        return {'component_id': 'pdf', 'file_path': file_path}
    elif file_type and file_type.startswith('text/xml'):
        return None
    else:
        return {'component_id': 'other', 'file_path': file_path, 'ftype': file_type}