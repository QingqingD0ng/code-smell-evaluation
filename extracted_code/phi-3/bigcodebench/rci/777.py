import re

import os

import zipfile

import logging


# Compile the regex pattern for performance

zip_pattern = re.compile(r'^(.*?)-\d+\.zip$')


def extract_zip_files(directory, pattern=zip_pattern):

    extracted_dirs = []

    compiled_pattern = pattern


    for filename in os.listdir(directory):

        if compiled_pattern.match(filename):

            zip_path = os.path.join(directory, filename)

            match = compiled_pattern.match(filename)

            prefix = match.group(1)

            extract_dir = os.path.normpath(os.path.join(directory, prefix))


            if not os.path.exists(extract_dir):

                os.makedirs(extract_dir)


            try:

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:

                    zip_ref.extractall(extract_dir)

                extracted_dirs.append(extract_dir)

                logging.info(f'Extracted {filename} to {extract_dir}')


            except (zipfile.BadZipFile, zipfile.LargeZipFile, IOError) as e:

                logging.error(f'Failed to extract {filename}: {e}')


    return extracted_dirs


# Example usage

extracted_directories = extract_zip_files('/tmp/my_data')