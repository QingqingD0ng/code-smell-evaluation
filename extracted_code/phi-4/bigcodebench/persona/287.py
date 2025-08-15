from collections import Counter
import os
import json

def task_func(filename, directory):
    total_words_count = Counter()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    words = content.split()
                    total_words_count.update(words)
    
    with open(filename, 'w') as f:
        json.dump(dict(total_words_count), f)
    
    return sum(total_words_count.values())