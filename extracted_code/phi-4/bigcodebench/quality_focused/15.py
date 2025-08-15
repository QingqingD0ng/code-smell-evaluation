import subprocess
import csv
import os

def task_func(commands_file_path, output_dir_path):
    if not os.path.exists(commands_file_path):
        raise FileNotFoundError(f"File not found: {commands_file_path}")

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_files = []

    with open(commands_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if not row:
                continue
            command = row[0]
            output_file_path = os.path.join(output_dir_path, f"command_{i + 1}_output.txt")
            with open(output_file_path, "w") as output_file:
                try:
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    output_file.write(result.stdout)
                    if result.returncode!= 0:
                        output_file.write(f"\nError: {result.stderr}\nExit code: {result.returncode}")
                except Exception as e:
                    output_file.write(f"Exception occurred: {str(e)}")
            output_files.append(output_file_path)

    return output_files