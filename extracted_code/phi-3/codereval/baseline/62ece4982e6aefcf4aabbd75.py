import subprocess

def addignored(ignored):
    # Run git command to find ignored files
    result = subprocess.run(['git', 'ls-files', '--others', '--ignored', '--exclude-standard'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if the command was successful
    if result.returncode!= 0:
        raise Exception("Git command failed: " + result.stderr)
    
    # Split the output by newlines to get individual file names
    files = result.stdout.strip().split('\n')
    
    # Sort the list of ignored file names
    files.sort()
    
    # Join the file names into a single string separated by commas
    return ', '.join(files)

# Example usage (uncomment to run):
# ignored_files = addignored()
# print(ignored_files)