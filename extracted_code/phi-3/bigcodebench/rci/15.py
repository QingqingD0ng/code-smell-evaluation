import csv
import os
import subprocess

def execute_commands_from_csv_and_save_outputs(csv_file_path, output_directory_path):
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"{csv_file_path} does not exist.")
    os.makedirs(output_directory_path, exist_ok=True)
    output_files_paths = []
    with open(csv_file_path, newline='') as csvfile:
        command_reader = csv.reader(csvfile)
        for index, row in enumerate(command_reader, start=1):
            command = row[0]
            output_file_name = f"command_{index}_output.txt"
            output_file_path = os.path.join(output_directory_path, output_file_name)
            try:
                process = subprocess.run(command, text=True, capture_output=True, shell=False)
                with open(output_file_path, 'w') as output_file:
                    output_file.write(process.stdout)
            except subprocess.CalledProcessError as e:
                with open(output_file_path, 'a') as output_file:
                    output_file.write(f"Command failed with exit code {e.returncode}:\n")
                    output_file.write(f"{e.stderr}\n")
                    output_file.write(f"{e.stdout}\n")
            output_files_paths.append(output_file_path)
    return output_files_paths