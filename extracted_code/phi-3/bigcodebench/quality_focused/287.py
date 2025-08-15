from collections import Counter
import os
import json

def task_func(filename, directory):
    word_count = Counter()
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                word_count.update(file.read().lower().split())
    with open(filename, 'w') as json_file:
        json.dump(word_count, json_file)
    return sum(word_count.values())