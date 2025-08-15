import subprocess

def addignored(ignored):
    subprocess.run(["cd", ignored], shell=True)
    result = subprocess.run(["git", "status", "--ignored", "--ignored-match", ".ignored"], capture_output=True, text=True)
    ignored_files = result.stdout.strip().split('\n')
    ignored_files = sorted(file for file in ignored_files if not file.endswith(".gitignore"))
    return ','.join(ignored_files)