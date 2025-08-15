def append_text_to_file(file_name, text_buffer, encoding, overwrite=False):

    mode = 'w' if overwrite else 'a'

    with open(file_name, mode, encoding=encoding) as file:

        file.write(text_buffer)