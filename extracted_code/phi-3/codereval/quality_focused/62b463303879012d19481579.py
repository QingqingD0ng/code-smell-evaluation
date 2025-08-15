def extract_number_and_supplement_from_issue(issue):
    number = issue.get('number')
    supplement = issue.get('supplement')
    return number, supplement