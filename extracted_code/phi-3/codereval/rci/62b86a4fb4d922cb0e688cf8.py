from typing import Union
import re

def validate_value(input_value: str, regex: str) -> bool:
    """
    Validates an input value against a regular expression.

    :param input_value: The string to validate.
    :param regex: A regular expression pattern.
    :return: True if the input_value matches the regex, False otherwise.
    """
    return bool(re.match(regex, input_value))