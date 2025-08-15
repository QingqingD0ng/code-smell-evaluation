import os

def append_text_to_file(file_name, text_buffer, encoding='utf-8', overwrite=False):
    mode = 'w' if overwrite else 'a'
    mode += '+' if not overwrite else ''
    with open(file_name, mode, encoding=encoding) as file:
        file.write(text_buffer)