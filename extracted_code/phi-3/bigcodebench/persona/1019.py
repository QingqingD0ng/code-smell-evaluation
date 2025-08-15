from PIL import Image
import codecs
import pytesseract

IMAGE_PATH = "image.png"

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    try:
        # Open the image file
        image = Image.open(filename)
        # Perform OCR to extract text
        text = pytesseract.image_to_string(image, lang='rus')
        # Decode the text using the original encoding
        decoded_text = codecs.decode(text, from_encoding)
        # Encode the text using the target encoding
        encoded_text = codecs.encode(decoded_text, to_encoding)
        return encoded_text
    except (UnicodeDecodeError, LookupError):
        # If OCR or encoding fails, attempt to extract comment from image
        try:
            # Assume a function to extract comment from image exists
            comment = extract_comment_from_image(filename)
            if comment:
                # Decode and encode the comment
                decoded_comment = codecs.decode(comment, from_encoding)
                encoded_comment = codecs.encode(decoded_comment, to_encoding)
                return encoded_comment
            else:
                return ""
        except (UnicodeDecodeError, LookupError):
            return ""

# Placeholder function for extracting comment from image
def extract_comment_from_image(filename):
    # This function should implement comment extraction logic
    return ""