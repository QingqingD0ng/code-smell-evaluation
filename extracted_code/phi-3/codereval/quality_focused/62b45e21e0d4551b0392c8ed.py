def find_path_type(path):
    import os
    import mimetypes

    # Check if the path is a directory
    if os.path.isdir(path):
        return 'Directory'

    # Check if the path is a file
    elif os.path.isfile(path):
        # Get the MIME type of the file
        mime_type, _ = mimetypes.guess_type(path)
        return mime_type if mime_type else 'Unknown File'

    else:
        return 'Invalid Path'