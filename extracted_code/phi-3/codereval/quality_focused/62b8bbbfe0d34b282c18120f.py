import codecs

def file_to_textbuffer(file_name, encoding):
    with codecs.open(file_name, 'r', encoding=encoding) as file:
        return file.read()