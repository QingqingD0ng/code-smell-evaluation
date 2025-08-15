import subprocess

def addignored():
    result = subprocess.run(['git', 'ls-files', '--others', '--ignored', '--exclude-standard'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return ', '.join(sorted(result.stdout.strip().split('\n')))

ignored_files = addignored()
print(ignored_files)