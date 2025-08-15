from PIL import Image
import codecs
import pytesseract

IMAGE_PATH = "image.png"

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    try:
        # Extract text using OCR
        text = pytesseract.image_to_string(Image.open(filename))
        # Convert encoding
        return text.encode(from_encoding).decode(to_encoding)
    except (UnicodeDecodeError, LookupError):
        try:
            # Attempt to read image comment
            image = Image.open(filename)
            comment = image.info.get("comment")
            if comment:
                # Convert comment encoding
                return comment.encode(from_encoding).decode(to_encoding)
        except (UnicodeDecodeError, LookupError):
            pass
    return ""

# Example usage
# text = task_func('image.png', 'cp1251', 'utf8')
# print(text)