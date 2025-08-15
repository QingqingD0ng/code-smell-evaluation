def test_tag(tag: str) -> bool:
    LEEGAL_TAG_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return all(char in LEEGAL_TAG_CHARS for char in tag)

This function takes a string `tag` as input and returns a boolean value. It uses a set named `LEEGAL_TAG_CHARS` that contains lowercase and uppercase alphabets and digits. The function checks if every character in the input `tag` is present in the set `LEEGAL_TAG_CHARS`. If any character is not present, the `all` function returns `False`. Otherwise, it returns `True`. This implementation avoids using any type of documentation, such as docstrings or comments, and is designed to be clean and efficient.