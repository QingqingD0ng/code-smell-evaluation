import subprocess
import csv
import os

def task_func(commands_file_path, output_dir_path):
    if not os.path.exists(commands_file_path):
        raise FileNotFoundError(f"The file {commands_file_path} does not exist.")

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_files = []

    with open(commands_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader, start=1):
            command = row[0]
            output_file_path = os.path.join(output_dir_path, f'command_{index}_output.txt')
            output_files.append(output_file_path)

            with open(output_file_path, 'w') as output_file:
                try:
                    result = subprocess.run(command, shell=True, check=True, text=True,
                                            stdout=output_file, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    output_file.write(f"Command failed with exit code {e.returncode}\n")
                    output_file.write(f"Error output:\n{e.output}\n")

    return output_files