def verifyClass(iface, candidate):

    if not hasattr(candidate, '__call__'):

        return False

    if not iface(candidate()):

        return False

    return True