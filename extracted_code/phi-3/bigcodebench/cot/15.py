import subprocess
import csv
import os

def task_func(commands_file_path, output_dir_path):
    if not os.path.exists(commands_file_path):
        raise FileNotFoundError(f"{commands_file_path} does not exist.")

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_files = []
    with open(commands_file_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            command = row[0]
            output_file_path = os.path.join(output_dir_path, f'command_{index}_output.txt')
            with open(output_file_path, 'w') as output_file:
                try:
                    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                    output_file.write(result.stdout)
                except subprocess.CalledProcessError as e:
                    output_file.write(f"Error executing command: {e.stderr}\nExit code: {e.returncode}")
            output_files.append(output_file_path)

    return output_files