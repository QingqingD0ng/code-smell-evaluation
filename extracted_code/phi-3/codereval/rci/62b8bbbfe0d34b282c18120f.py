import io

def file_to_textbuffer(file_name, encoding):
    try:
        with open(file_name, 'r', encoding=encoding) as file:
            return io.TextIOWrapper(file.buffer, encoding=encoding)
    except FileNotFoundError:
        raise Exception(f"File '{file_name}' not found.")
    except UnicodeDecodeError:
        raise Exception(f"Encoding '{encoding}' not supported or file is corrupted.")