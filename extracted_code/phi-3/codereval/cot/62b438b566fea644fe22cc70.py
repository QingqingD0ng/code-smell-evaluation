import argparse
import inspect
import subprocess

def get_command_help(command_name):
    process = subprocess.Popen(
        ['borg', 'help', command_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    if process.returncode!= 0:
        raise Exception(f"Error getting help for '{command_name}': {stderr}")
    return stdout

def extract_arguments(help_text):
    lines = [line for line in help_text.split('\n') if line.strip()]
    argument_lines = [line for line in lines if line.startswith('  option')]
    arguments = {}
    for line in argument_lines:
        parts = line.split()
        arg_name = parts[1][2:]
        arg_help = ''.join(parts[2:])
        arguments[arg_name] = arg_help
    return arguments

def generate_bash_completion(borg_commands):
    completion_script = "function _borgmatic_completion() {\n"
    for command in borg_commands:
        help_text = get_command_help(command)
        arguments = extract_arguments(help_text)
        completion_script += f"    _borgmatic_complete \"{command}\"\n"
        for arg, description in arguments.items():
            completion_script += f"    _borgmatic_complete_argument \"{arg}\" \"{description}\"\n"
    completion_script += "}\n"
    return completion_script

if __name__ == "__main__":
    borg_commands = ['create', 'list','remotes']  # Example list of borgmatic commands
    print(generate_bash_completion(borg_commands))