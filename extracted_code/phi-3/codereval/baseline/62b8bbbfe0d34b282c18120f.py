import io

def file_to_textbuffer(file_name, encoding):
    with open(file_name, 'r', encoding=encoding) as file:
        return io.TextIOWrapper(file.buffer)