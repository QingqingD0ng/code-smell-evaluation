import unicodedata
import csv
from collections import Counter
import matplotlib.pyplot as plt

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def task_func(csv_file):
    try:
        with open(csv_file, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            text =''.join(' '.join(row) for row in reader)
            normalized_text = normalize_text(text)
            words = normalized_text.split()
            word_counts = Counter(words)
            most_common_words = word_counts.most_common(10)
            
            words, frequencies = zip(*most_common_words)
            
            plt.figure(figsize=(10, 6))
            ax = plt.bar(words, frequencies)
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title('Top 10 Most Common Words')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            return ax, most_common_words
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {csv_file}")
    except IOError:
        raise IOError(f"Error reading file: {csv_file}")