def match(filename):
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    return filename.lower() == 'doxyfile'