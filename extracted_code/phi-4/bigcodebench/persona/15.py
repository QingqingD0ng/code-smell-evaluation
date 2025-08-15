import subprocess
import csv
import os

def task_func(commands_file_path, output_dir_path):
    if not os.path.exists(commands_file_path):
        raise FileNotFoundError(f"The file {commands_file_path} does not exist.")
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_files = []
    with open(commands_file_path, mode='r', newline='') as csvfile:
        command_reader = csv.reader(csvfile)
        for index, row in enumerate(command_reader, start=1):
            command = row[0]
            output_file_path = os.path.join(output_dir_path, f'command_{index}_output.txt')
            output_files.append(output_file_path)

            with open(output_file_path, 'w') as output_file:
                try:
                    result = subprocess.run(command, shell=True, check=True, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    output_file.write(result.stdout)
                except subprocess.CalledProcessError as e:
                    error_message = (f"Command '{command}' failed with exit code {e.returncode}.\n"
                                     f"Error Output:\n{e.stderr}")
                    output_file.write(error_message)

    return output_files