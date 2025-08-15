import re


def _extract_number_and_supplement_from_issue_element(issue):

    # Regular expression to extract number and supplement from the issue element

    pattern = r'(\d+)(?:-\d+)?'

    matches = re.findall(pattern, issue)

    number = matches[0] if matches else None

    supplement = '-' + matches[1] if len(matches) > 1 else None

    return number, supplement