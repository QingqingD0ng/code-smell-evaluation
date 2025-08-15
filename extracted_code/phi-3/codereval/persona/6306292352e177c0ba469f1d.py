import re

from typing import Callable, Set, Tuple


def find_tags(text: str, replacer: Callable = None) -> Tuple[Set[str], str]:

    tags = set()

    in_code_block = False

    result_text = []


    for match in re.finditer(r'(@\w+)', text):

        tag = match.group(0)

        start, end = match.span()


        if text[start - 1] == '`' and (in_code_block or not in_code_block):

            # Check if we are starting or ending a code block

            if text[start - 2] == '`' and text[end] == '`':

                in_code_block = not in_code_block


        if not in_code_block:

            tags.add(tag)

            if replacer:

                replacement = replacer(tag[1:-1])

                text = text[:start] + replacement + text[end:]


        result_text.append(text[start:end])


    return tags, ''.join(result_text)