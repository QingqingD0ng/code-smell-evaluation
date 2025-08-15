from typing import List, Optional

def int_to_string(number: int, alphabet: List[str], padding: Optional[int] = None) -> str:
    if padding is not None:
        number = number % (alphabet_length ** padding)
    
    result = []
    while number > 0:
        number, remainder = divmod(number, alphabet_length)
        result.append(alphabet[remainder])
    
    return ''.join(reversed(result))