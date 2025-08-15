def plus_or_dot(pieces):
    return '.' if pieces[-1].endswith('+') else '+'

pieces = ["text", "+extra"]
print(plus_or_dot(pieces))  # Output: '.'

pieces = ["text"]
print(plus_or_dot(pieces))  # Output: '+'

pieces = ["text+"]
print(plus_or_dot(pieces))  # Output: '.'