from typing import List


def string_to_int(string: str, alphabet: List[str]) -> int:

    base = len(alphabet)

    result = 0

    for char in string:

        result = result * base + alphabet.index(char)

    return result