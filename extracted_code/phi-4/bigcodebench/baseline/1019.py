from PIL import Image
import codecs
import pytesseract

IMAGE_PATH = "image.png"

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    try:
        image = Image.open(filename)
        extracted_text = pytesseract.image_to_string(image).encode(from_encoding).decode(to_encoding)
        return extracted_text
    except (UnicodeDecodeError, LookupError):
        try:
            with open(filename, "rb") as f:
                f.seek(0)
                if f.read(2) == b'\xff\xd8':  # JPEG files
                    f.seek(-128, 2)
                    segments = f.read().split(b'\xff\xfe')
                    for segment in segments:
                        if b'Comment' in segment:
                            comment = segment.split(b'\x00', 1)[1].split(b'\x00', 1)[0]
                            return comment.decode(from_encoding).encode(to_encoding).decode(to_encoding)
        except Exception:
            pass
    return ""