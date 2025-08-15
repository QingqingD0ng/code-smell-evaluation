LEEGAL_TAG_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"

def test_tag(tag: str) -> bool:
    return all(char in LEEGAL_TAG_CHARS for char in tag)