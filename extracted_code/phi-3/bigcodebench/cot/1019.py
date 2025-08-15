from PIL import Image
import codecs
import pytesseract

IMAGE_PATH = "image.png"

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    image = Image.open(filename)
    image_text = pytesseract.image_to_string(image, lang='rus')
    try:
        return codecs.encode(image_text.encode(from_encoding), to_encoding)
    except (codecs.UnicodeDecodeError, LookupError):
        # Fallback to image comment processing (not implemented in this snippet)
        # For demonstration, assume there's a function `extract_comment_from_image`
        # comment = extract_comment_from_image(image)
        # try:
        #     return codecs.encode(comment.encode(from_encoding), to_encoding)
        # except (codecs.UnicodeDecodeError, LookupError):
        return ""