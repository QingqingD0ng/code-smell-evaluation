import typing

def int_to_string(number: int, alphabet: list, padding: typing.Optional[int] = None) -> str:
    result = []
    base = len(alphabet)
    while number > 0:
        number, remainder = divmod(number - 1, base)
        result.append(alphabet[remainder])
    result.reverse()
    
    if padding is not None:
        result = ['0'] * padding + result
    return ''.join(result)