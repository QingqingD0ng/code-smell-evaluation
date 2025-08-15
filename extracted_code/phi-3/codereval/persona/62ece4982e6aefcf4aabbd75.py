import subprocess
import re

def add_ignored(ignored):
    # Run the git command to get the list of ignored files
    result = subprocess.run(
        ['git', 'ls-files', '--others', '--ignored', '--exclude-standard'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Check if the command was successful
    if result.returncode!= 0:
        raise Exception(f"Git command failed: {result.stderr}")
    
    # Extract file names using a regular expression
    file_names = re.findall(r'^\w+\.\w+$', result.stdout, re.MULTILINE)
    
    # Sort the list of file names
    file_names.sort()
    
    # Join the file names into a single string separated by commas
    return ','.join(file_names)