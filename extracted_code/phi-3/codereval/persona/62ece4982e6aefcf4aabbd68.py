import typing

def int_to_string(number: int, alphabet: typing.List[str], padding: typing.Optional[int] = None) -> str:
    result = ''
    while number > 0:
        result = alphabet[number % len(alphabet)] + result
        number //= len(alphabet)
    if padding is not None:
        result = result.rjust(padding, alphabet[0])
    return result