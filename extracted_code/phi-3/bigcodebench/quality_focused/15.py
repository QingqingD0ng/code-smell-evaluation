import subprocess
import csv
import os

def task_func(commands_file_path, output_dir_path):
    if not os.path.exists(commands_file_path):
        raise FileNotFoundError(f"{commands_file_path} does not exist.")
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    output_files_paths = []
    
    with open(commands_file_path, newline='') as csvfile:
        command_reader = csv.reader(csvfile)
        for index, row in enumerate(command_reader, start=1):
            command = row[0]
            output_file_name = f"command_{index}_output.txt"
            output_file_path = os.path.join(output_dir_path, output_file_name)
            output_files_paths.append(output_file_path)
            
            try:
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                with open(output_file_path, 'w') as outfile:
                    outfile.write(result.stdout.decode())
            except subprocess.CalledProcessError as e:
                error_message = f"Command failed with exit code {e.returncode}: {e.stderr.decode()}"
                with open(output_file_path, 'a') as outfile:
                    outfile.write(error_message)
    
    return output_files_paths

# Example usage:
# output_files = task_func("commands.csv", "/path/to/output_directory")