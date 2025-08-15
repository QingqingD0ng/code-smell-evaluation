import subprocess
import shlex

def add_ignored(ignored):
    # Execute the git command to list ignored files and capture the output
    result = subprocess.run(shlex.split(f"git ls-files -i --exclude-standard --others"), capture_output=True, text=True)
    # Split the output by newlines to get a list of file names
    file_names = result.stdout.splitlines()
    # Filter out empty lines and sort the list
    sorted_ignored_files = sorted(filter(None, file_names))
    # Join the sorted list into a single string with commas
    return ','.join(sorted_ignored_files)