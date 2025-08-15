import re


def oneline(script, separator=" && "):
    commands = re.split(r'(?<!\w)([;,\(\)]|\"[^\"]*\")', script)
    return separator.join(filter(None, commands))

# Example usage:

script = 'command1; "multi-arg command 2" && command3 (command4) && command5'
print(oneline(script))