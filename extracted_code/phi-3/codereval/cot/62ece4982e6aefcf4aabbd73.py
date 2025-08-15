import platform
import shlex

def split(s, platform='this'):
    if platform == 'this':
        platform = platform.system().lower()
    if platform == 'posix':
        return shlex.split(s)
    elif platform == 'windows' or platform == 'cmd':
        return shlex.split(s, posix=False)
    else:
        raise ValueError("Invalid platform specified")

input_string = "This is a test string with spaces."
print(split(input_string, 'posix'))
print(split(input_string, 'windows'))