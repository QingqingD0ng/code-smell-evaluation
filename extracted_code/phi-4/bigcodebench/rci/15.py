import subprocess
import csv
import os

def task_func(commands_file_path, output_dir_path):
    if not os.path.exists(commands_file_path):
        raise FileNotFoundError(f"The file {commands_file_path} does not exist.")
    
    os.makedirs(output_dir_path, exist_ok=True)
    
    output_files = []
    
    with open(commands_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader, start=1):
            if not row:
                continue
            command = row[0]
            output_file_path = os.path.join(output_dir_path, f"command_{index}_output.txt")
            try:
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                with open(output_file_path, 'w') as output_file:
                    output_file.write(result.stdout)
            except subprocess.CalledProcessError as e:
                with open(output_file_path, 'w') as output_file:
                    output_file.write(f"Command failed with exit code {e.returncode}:\n")
                    output_file.write(e.stderr)
            except Exception as e:
                output_file_path = os.path.join(output_dir_path, f"command_{index}_error.txt")
                with open(output_file_path, 'w') as output_file:
                    output_file.write(f"Unexpected error occurred: {e}")
            output_files.append(output_file_path)
    
    return output_files