import sys

def tzname_in_python2_refined(namefunc):
    if sys.version_info[0] < 3:
        # Ensure we only encode when dealing with a Unicode string
        if isinstance(namefunc(), str):
            return namefunc().encode('utf-8')
        else:
            return namefunc()
    return namefunc()