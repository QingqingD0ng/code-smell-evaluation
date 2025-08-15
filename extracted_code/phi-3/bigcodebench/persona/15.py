import subprocess
import csv
import os

def task_func(commands_file_path, output_dir_path):
    if not os.path.exists(commands_file_path):
        raise FileNotFoundError("commands_file_path does not exist")
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    output_paths = []
    command_idx = 1
    
    with open(commands_file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            command = row[0].strip()
            output_file_path = os.path.join(output_dir_path, f'command_{command_idx}_output.txt')
            output_paths.append(output_file_path)
            
            try:
                result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
                with open(output_file_path, 'w') as output_file:
                    output_file.write(result.stdout)
            except subprocess.CalledProcessError as e:
                with open(output_file_path, 'a') as output_file:
                    error_message = f"Command failed with exit code {e.returncode}: {e.stderr}\n{e.stdout}"
                    output_file.write(error_message)
            
            command_idx += 1
            
    return output_paths