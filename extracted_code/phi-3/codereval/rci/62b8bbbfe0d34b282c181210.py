import os


def append_text_to_file(file_name, text_buffer, encoding='utf-8', overwrite=False):
    if not file_name.endswith('.txt'):
        raise ValueError("File extension must be.txt for overwrite feature.")
    
    mode = 'w' if overwrite else 'a'
    with open(file_name, mode, encoding=encoding) as file:
        file.write(text_buffer)