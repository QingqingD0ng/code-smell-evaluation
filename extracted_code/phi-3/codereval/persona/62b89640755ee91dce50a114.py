import sys

if sys.version_info[0] < 3:
    def tzname_in_python2(namefunc):
        return namefunc().encode('utf-8')
else:
    def tzname_in_python2(namefunc):
        return namefunc()