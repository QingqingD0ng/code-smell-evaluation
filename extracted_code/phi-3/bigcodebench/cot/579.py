import unicodedata
import csv
from collections import Counter
import matplotlib.pyplot as plt

def task_func(csv_file):
    word_counts = Counter()
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                text = ''.join(row)
                text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
                words = text.split()
                word_counts.update(words)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {csv_file} not found.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file {csv_file}: {e}")

    most_common_words = word_counts.most_common(10)
    fig, ax = plt.subplots()
    ax.bar([word for word, _ in most_common_words], [count for _, count in most_common_words])
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('10 Most Common Words')
    plt.xticks(rotation=45)
    return ax, most_common_words