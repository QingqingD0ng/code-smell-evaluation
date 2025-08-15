import subprocess

import os

import sys

import glob


def task_func(directory_path):

    bat_files = glob.glob(os.path.join(directory_path, "*.bat"))

    results = []


    for bat_file in bat_files:

        try:

            result = subprocess.run(bat_file, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            results.append((os.path.basename(bat_file), result.returncode))

        except subprocess.CalledProcessError as e:

            results.append((os.path.basename(bat_file), None))


    return results