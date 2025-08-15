from PIL import Image
import codecs
import pytesseract

IMAGE_PATH = "image.png"

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    try:
        # Open the image file
        image = Image.open(filename)
        
        # Try to extract text using OCR
        try:
            extracted_text = pytesseract.image_to_string(image)
            comment = codecs.decode(extracted_text.encode(from_encoding), to_encoding)
            return comment
        except (UnicodeDecodeError, LookupError):
            pass
        
        # Fallback to image comment processing
        try:
            if "comment" in image.info:
                comment = image.info["comment"]
                return codecs.decode(comment.encode(from_encoding), to_encoding)
        except (UnicodeDecodeError, LookupError):
            pass
        
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
    
    # Return an empty string if both OCR and comment processing fail
    return ""