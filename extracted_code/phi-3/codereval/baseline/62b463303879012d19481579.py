def _extract_number_and_supplment_from_issue_element(issue):
    number = None
    supplment = None
    
    for content in issue:
        if content.get('type') == 'number':
            number = content.get('value')
        elif content.get('type') =='supplment':
            supplment = content.get('value')
    
    return number, supplment