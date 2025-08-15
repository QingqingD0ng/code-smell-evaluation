import subprocess

def bash_completion():
    # List of known commands (should be extended based on actual parser output)
    commands = ['start','stop','status','restart', 'list','remove']
    
    # Get the help output from borgmatic
    process = subprocess.Popen(['borgmatic', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Check if borgmatic's help command was successful
    if process.returncode!= 0:
        raise Exception(f"Error running borgmatic --help: {stderr.decode()}")
    
    # Parse the help output to extract the available subcommands
    help_output = stdout.decode()
    subcommands = []
    for line in help_output.split('\n'):
        if 'Usage:' in line:
            parts = line.split()
            if parts and parts[0] == 'Usage:':
                subcommands.extend(parts[1].split()[1:])
    
    # Create completion script content
    completion_script = "__borgmatic_commands=" +''.join(commands + subcommands) + "\n"
    completion_script += "\n"
    for cmd in commands:
        completion_script += f"complete -F _borgmatic_complete '{cmd}'\n"
        completion_script += f"_borgmatic_complete()\n"
        completion_script += "{\n"
        completion_script += "    local cur prev opts cur_opt prev_opt=\'\'\n"
        completion_script += "    COMPREPLY=()\n"
        completion_script += "    _get_comp_words_by_ref cur prev opts cur_opt prev_opt\n"
        completion_script += "    case \"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\${cur}#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\