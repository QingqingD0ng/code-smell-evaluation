import os
import mimetypes

def _eval_file(prefix, file_path):
    # Determine the file type
    file_type, _ = mimetypes.guess_type(file_path)
    if file_type is None:
        return None

    # Check if the file type is XML
    if file_type == 'application/xml':
        return None

    # If the file has the given prefix in its name
    if prefix in os.path.basename(file_path):
        # Return a dictionary with the file path
        return {'file_path': file_path}
    else:
        # Return a dictionary with the file path and file type
        return {'file_path': file_path, 'ftype': file_type}