from collections import Counter
import os
import json

def task_func(filename, directory):
    word_count = Counter()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    word_count.update(f.read().lower().split())
    with open(os.path.join(directory, filename), 'w') as json_file:
        json.dump(word_count, json_file)
    return sum(word_count.values())