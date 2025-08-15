class FillRequestEligible:
    def __init__(self):
        self.fill = None
        self.request = None

class FillRequestNotEligible:
    def __init__(self):
        self.fill = None
        self.request = None

def determine_fill_request_eligibility(obj):
    if isinstance(obj, FillRequestEligible):
        return True
    elif isinstance(obj, FillRequestNotEligible):
        return False
    else:
        raise TypeError("Object must be an instance of FillRequestEligible or FillRequestNotEligible")

eligible_obj = FillRequestEligible()
not_eligible_obj = FillRequestNotEligible()

print(determine_fill_request_eligibility(eligible_obj))  # Output: True
print(determine_fill_request_eligibility(not_eligible_obj))  # Output: False