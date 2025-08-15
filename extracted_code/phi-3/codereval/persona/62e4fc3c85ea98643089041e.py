import subprocess

def _inline_r_setup(code: str) -> str:
    # Start R process and pass the initialization code
    process = subprocess.Popen(['R', '--slave'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Write the initialization code to the R process
    stdout, stderr = process.communicate(input=code.encode('utf-8'))
    
    # Check for errors
    if stderr:
        raise Exception(f"Error initializing R: {stderr.decode('utf-8')}")
    
    # Return the R output
    return stdout.decode('utf-8')

# Example usage
r_code = """
library(stats)
options(digits=5)
"""

output = _inline_r_setup(r_code)
print(output)