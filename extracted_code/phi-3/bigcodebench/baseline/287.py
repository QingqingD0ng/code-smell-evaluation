import os
import json
from collections import Counter

def task_func(filename, directory):
    word_count = Counter()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    words = text.split()
                    word_count.update(words)
    
    total_words = sum(word_count.values())
    
    with open(os.path.join(directory, filename), 'w') as json_file:
        json.dump(word_count, json_file)
        
    return total_words

# Example usage (uncomment and run in a suitable environment):
# if __name__ == "__main__":
#     print(task_func('word_count.json', './testdir/'))