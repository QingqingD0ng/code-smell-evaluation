class TextBuffer:
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)

def file_to_textbuffer(file_name, encoding):
    text_buffer = TextBuffer()
    with open(file_name, 'r', encoding=encoding) as file:
        for line in file:
            text_buffer.append(line)
    return text_buffer