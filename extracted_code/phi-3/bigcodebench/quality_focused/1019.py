from PIL import Image
import pytesseract
import codecs

IMAGE_PATH = "image.png"

def extract_text_from_image(image_path, encoding):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='rus')
    return text

def extract_comment_from_image(image_path, encoding):
    with Image.open(image_path) as img:
        # Assuming comments are in the Image info metadata under 'comment'
        comment = img.info.get('comment', '')
        return codecs.decode(comment, encoding)

def process_image(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    try:
        text = extract_text_from_image(filename, from_encoding)
        comment = extract_comment_from_image(filename, from_encoding)
        combined_text = text + "\n" + comment
        return codecs.encode(combined_text, to_encoding)
    except (UnicodeDecodeError, LookupError):
        return ''