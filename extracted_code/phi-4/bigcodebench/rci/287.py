from collections import Counter
import os
import json

def task_func(filename, directory, extension='.txt'):
    total_word_count = 0
    word_counter = Counter()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            words = line.split()
                            word_counter.update(words)
                            total_word_count += len(words)
                except IOError:
                    pass

    try:
        with open(filename, 'w') as json_file:
            json.dump(word_counter, json_file, indent=4, sort_keys=True)
    except IOError:
        pass

    return total_word_count