import subprocess
import re

def bash_completion():
    commands = ['start','stop','status','restart', 'list','remove']
    process = subprocess.Popen(['borgmatic', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode!= 0:
        raise Exception(f"Error running borgmatic --help: {stderr.decode().strip()}")

    help_output = stdout.decode().strip()
    subcommands = re.findall(r'^\s+Usage: borgmatic\s+(\w+)', help_output, re.MULTILINE)
    commands.extend(subcommands)

    completion_script = f"__borgmatic_commands='{' '.join(commands)}'\n"
    completion_script += "\n"
    completion_script += "complete -F _borgmatic_complete 'borgmatic'\n"
    completion_script += "_borgmatic_complete()\n"
    completion_script += "{\n"
    completion_script += "    local cur prev opts cur_opt prev_opt=\"\"\n"
    completion_script += "    COMPREPLY=()\n"
    completion_script += "    _get_comp_words_by_ref cur prev opts cur_opt prev_opt\n"
    completion_script += "    case \"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\