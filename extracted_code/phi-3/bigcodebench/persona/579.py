import unicodedata
import csv
from collections import Counter
import matplotlib.pyplot as plt

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def task_func(csv_file):
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            text_data =''.join(normalize_text(row[0]) for row in reader if row)

        word_counts = Counter(text_data.split())
        most_common_words = word_counts.most_common(10)

        fig, ax = plt.subplots()
        ax.bar(range(len(most_common_words)), [count for word, count in most_common_words], align='center')
        ax.set_xticks(range(len(most_common_words)))
        ax.set_xticklabels([word for word, count in most_common_words])
        plt.show()

        return ax, most_common_words

    except FileNotFoundError:
        raise FileNotFoundError(f"File {csv_file} not found.")
    except IOError as e:
        raise IOError(f"An error occurred while reading the file: {e}")