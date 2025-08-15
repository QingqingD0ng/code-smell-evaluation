from PIL import Image
import codecs
import pytesseract

IMAGE_PATH = "image.png"

def task_func(filename=IMAGE_PATH, from_encoding="cp1251", to_encoding="utf8"):
    try:
        # Open the image
        img = Image.open(filename)
        
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(img, lang='rus')
        
        # Decode the extracted text
        decoded_text = text.encode(from_encoding).decode(to_encoding)
        
        return decoded_text
    except (UnicodeDecodeError, LookupError):
        # Fallback to image comment if OCR fails, assuming image has a comment attribute
        try:
            with open(filename, 'rb') as img_file:
                img_data = img_file.read()
                # Assuming a function 'extract_comment' that gets the comment from the image
                comment = extract_comment(img_data)
                # Decode the extracted comment
                decoded_comment = comment.encode(from_encoding).decode(to_encoding)
                return decoded_comment
        except Exception as e:
            return ""
        return ""

# Assuming a function 'extract_comment' exists that extracts comments from image data
def extract_comment(img_data):
    # Placeholder for actual comment extraction logic
    pass