from PIL import Image
import codecs
import pytesseract
import logging
from some_metadata_library import extract_comment_from_metadata

IMAGE_PATH = "image.png"

def validate_parameters(filename, from_encoding, to_encoding):
    if not filename.endswith('.png'):
        raise ValueError("Invalid filename. Only.png files are supported.")
    if from_encoding not in ['cp1251', 'utf8'] or to_encoding not in ['cp1251', 'utf8']:
        raise ValueError("Invalid encoding. Supported encodings are 'cp1251' and 'utf8'.")

def extract_text_from_image(image_path, language='rus'):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang=language)

def extract_comment_from_image(image_path):
    # Placeholder for actual comment extraction logic based on image metadata
    # This is where you'd implement the logic to extract a comment from the image's metadata
    return extract_comment_from_metadata(image_path)

def convert_text_encoding(text, from_encoding, to_encoding):
    try:
        return text.encode(from_encoding).decode(to_encoding)
    except (UnicodeDecodeError, LookupError) as e:
        logging.error(f"Error converting text encoding: {e}")
        return ""

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    validate_parameters(filename, from_encoding, to_encoding)
    
    try:
        text = extract_text_from_image(filename)
        return convert_text_encoding(text, from_encoding, to_encoding)
    except Exception as e:
        logging.exception("OCR extraction failed. Falling back to image comment extraction.")
        try:
            comment = extract_comment_from_image(filename)
            return convert_text_encoding(comment, from_encoding, to_encoding)
        except Exception as e:
            logging