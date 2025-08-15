def _extract_number_and_supplment_from_issue_element(issue):
    number, supplment = None, None
    numbers, supplements = [], []
    
    for content in issue:
        if content.get('type') == 'number':
            number = content.get('value')
            numbers.append(number)
        elif content.get('type') =='supplment':
            supplement = content.get('value')
            supplements.append(supplement)
    
    return (numbers, supplements) if numbers or supplements else (None, None)