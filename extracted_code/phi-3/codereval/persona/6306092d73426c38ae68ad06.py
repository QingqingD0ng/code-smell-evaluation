silent_args = {}

for arg in args:
    if arg.startswith('--no-'):
        arg_name = arg[len('--no-'):]
        silent_args[arg_name] = True