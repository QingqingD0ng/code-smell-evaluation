import unicodedata
import csv
from collections import Counter
import matplotlib.pyplot as plt

def task_func(csv_file):
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            text = " ".join(" ".join(row) for row in reader)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {csv_file}")
    except IOError:
        raise IOError(f"Error reading file: {csv_file}")

    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    words = normalized_text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)

    words, counts = zip(*most_common_words)
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Most Common Words')

    return ax, most_common_words