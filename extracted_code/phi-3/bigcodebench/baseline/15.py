import subprocess
import csv
import os

def task_func(commands_file_path, output_dir_path):
    if not os.path.exists(commands_file_path):
        raise FileNotFoundError(f"{commands_file_path} does not exist.")
    os.makedirs(output_dir_path, exist_ok=True)
    output_files_paths = []
    with open(commands_file_path, newline='') as csvfile:
        command_reader = csv.reader(csvfile)
        for index, row in enumerate(command_reader, start=1):
            command = row[0]
            output_file_path = os.path.join(output_dir_path, f"command_{index}_output.txt")
            try:
                result = subprocess.run(command, text=True, capture_output=True, shell=True, check=True)
                with open(output_file_path, 'w') as output_file:
                    output_file.write(result.stdout)
            except subprocess.CalledProcessError as e:
                with open(output_file_path, 'a') as output_file:
                    output_file.write(f"Command failed with exit code {e.returncode}: {e.stderr}\n")
                    output_file.write(f"{e.stdout}\n")
            output_files_paths.append(output_file_path)
    return output_files_paths