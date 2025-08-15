import codecs

def file_to_textbuffer(file_name, encoding):
    with codecs.open(file_name, 'r', encoding=encoding) as file:
        text_buffer = file.read()
    return text_buffer