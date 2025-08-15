def verifyClass(iface, candidate):
    iface_methods = dir(iface)
    candidate_methods = dir(candidate)
    
    for method in iface_methods:
        if method not in candidate_methods:
            return False
    
    return True