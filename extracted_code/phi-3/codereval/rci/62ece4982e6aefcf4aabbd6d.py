from typing import List

def string_to_int(string: str, alphabet: List[str]) -> int:
    base = len(alphabet)
    result = 0
    for char in string:
        try:
            result = result * base + alphabet.index(char)
        except ValueError:
            raise ValueError(f"Character {char} not found in alphabet")
    return result