def test_tag(tag: str) -> bool:
    LEEGAL_TAG_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
    return all(char in LEEGAL_TAG_CHARS for char in tag)