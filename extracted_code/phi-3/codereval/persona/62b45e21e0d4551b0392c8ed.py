def find_path_type(path):
    import os
    import mimetypes

    # Extract the file extension from the path
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    # Get the MIME type associated with the file extension
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type.split('/')[-1] if mime_type else None

    # Determine the type based on MIME type
    if mime_type in ['text', 'plain']:
        return 'Text File'
    elif mime_type in ['html', 'xhtml']:
        return 'HTML File'
    elif mime_type in ['pdf', 'application/pdf']:
        return 'PDF File'
    elif mime_type in ['image/jpeg', 'image/png', 'image/gif', 'image/bmp']:
        return 'Image File'
    elif mime_type in ['video', 'application/x-msvideo', 'video/mp4', 'video/quicktime']:
        return 'Video File'
    elif mime_type in ['audio', 'audio/mpeg', 'audio/x-wav', 'audio/x-m4a']:
        return 'Audio File'
    elif mime_type in ['application/zip', 'application/x-zip-compressed']:
        return 'Compressed Archive'
    elif mime_type in ['application/octet-stream']:
        return 'Unknown Binary File'
    elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        return 'Excel File'
    elif mime_type in ['application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation']:
        return 'PowerPoint File'
    elif mime_type in ['application/vnd.ms-word', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return 'Word