import unicodedata
import csv
from collections import Counter
import matplotlib.pyplot as plt

def task_func(csv_file):
    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            text =''.join(' '.join(row) for row in reader)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {csv_file}")
    except IOError as e:
        raise IOError(f"Error reading file: {e}")

    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    words = normalized_text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)

    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 6))
    ax = plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Words')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return ax, most_common_words