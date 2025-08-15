from PIL import Image
import pytesseract
import codecs

IMAGE_PATH = "image.png"

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    try:
        img = Image.open(filename)
        text = pytesseract.image_to_string(img)
        comment = text.encode(from_encoding).decode(to_encoding)
    except (UnicodeDecodeError, LookupError):
        try:
            comment = img.info.get('comment', '')
            comment = comment.encode(from_encoding).decode(to_encoding)
        except (UnicodeDecodeError, LookupError) as e:
            raise ValueError("Incorrect encodings provided") from e
    return comment