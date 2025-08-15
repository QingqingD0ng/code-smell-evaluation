from collections import Counter
import os
import json

def task_func(filename, directory):
    word_counts = Counter()
    total_words = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    words = f.read().split()
                    word_counts.update(words)
                    total_words += len(words)

    with open(filename, 'w') as json_file:
        json.dump(word_counts, json_file)

    return total_words