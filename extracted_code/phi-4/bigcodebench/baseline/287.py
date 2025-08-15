from collections import Counter
import os
import json

def task_func(filename, directory):
    total_word_count = 0
    word_counter = Counter()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    words = content.split()
                    word_counter.update(words)
                    total_word_count += len(words)

    with open(filename, 'w') as json_file:
        json.dump(word_counter, json_file)

    return total_word_count