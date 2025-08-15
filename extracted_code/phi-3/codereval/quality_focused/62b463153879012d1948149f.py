import os
import mimetypes

def _eval_file(prefix, file_path):
    file_type, _ = mimetypes.guess_type(file_path)
    if file_type is None or file_type.startswith('application/xml'):
        return None

    if file_type == 'application/pdf':
        return {'component_id': prefix + '_' + os.path.basename(file_path), 'file_path': file_path}

    return {
        'component_id': prefix + '_' + os.path.basename(file_path),
        'ftype': file_type,
        'file_path': file_path
    }