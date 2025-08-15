from PIL import Image
import pytesseract

JPEG_HEADER = b'\xff\xd8'
JPEG_COMMENT_MARKER = b'\xff\xfeComment'

IMAGE_PATH = "image.png"

def extract_text_from_image(image_path, from_encoding, to_encoding):
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image).encode(from_encoding).decode(to_encoding)
        return extracted_text
    except (UnicodeDecodeError, LookupError):
        return None

def extract_comment_from_jpeg(image_path, from_encoding, to_encoding):
    try:
        with open(image_path, "rb") as f:
            if f.read(2) == JPEG_HEADER:
                f.seek(-128, 2)
                segments = f.read().split(JPEG_COMMENT_MARKER)
                for segment in segments:
                    if b'\x00' in segment:
                        comment = segment.split(b'\x00', 1)[1].split(b'\x00', 1)[0]
                        return comment.decode(from_encoding).encode(to_encoding).decode(to_encoding)
    except (UnicodeDecodeError, LookupError, IndexError):
        return None
    return None

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    text = extract_text_from_image(filename, from_encoding, to_encoding)
    if text:
        return text
    
    comment = extract_comment_from_jpeg(filename, from_encoding, to_encoding)
    if comment:
        return comment
    
    return ""