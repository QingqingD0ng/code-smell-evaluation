def _extract_number_and_supplment_from_issue_element(issue):
    number = issue.get('number')
    supplment = issue.get('suppliment')
    return number, supplment