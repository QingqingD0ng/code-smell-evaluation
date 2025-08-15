import math
from typing import List, Optional

def int_to_string(number: int, alphabet: List[str], padding: Optional[int] = None) -> str:
    base = len(alphabet)
    length = int(math.ceil(math.log(number + 1, base))) if padding is None else padding
    result = ''
    while number > 0:
        result = alphabet[number % base] + result
        number //= base
    return alphabet[0] * (length - len(result)) + result if padding else result