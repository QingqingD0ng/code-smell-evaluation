from collections import Counter
import os
import json

def task_func(filename, directory):
    total_words = 0
    word_count = Counter()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read().split()
                    total_words += len(content)
                    word_count.update(content)

    with open(filename, 'w') as f:
        json.dump(word_count, f)

    return total_words