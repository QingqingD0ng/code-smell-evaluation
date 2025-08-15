from typing import List, Optional

def int_to_shortuuid(number: int, alphabet: List[str], padding: Optional[int] = None) -> Optional[str]:
    if number < 0:
        return None
    base = len(alphabet)
    num_digits = int(math.ceil(math.log(number + 1, base))) if padding is None else padding
    uuid_str = ''
    while number > 0:
        uuid_str = alphabet[number % base] + uuid_str
        number //= base
    return alphabet[0] * (num_digits - len(uuid_str)) + uuid_str if padding else uuid_str