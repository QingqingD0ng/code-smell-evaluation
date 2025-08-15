import unicodedata
import csv
from collections import Counter
import matplotlib.pyplot as plt

def task_func(csv_file):
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            data =''.join([' '.join(row) for row in reader])
            ascii_data = unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf-8')
            words = ascii_data.split()
            word_counts = Counter(words)
            most_common_words = word_counts.most_common(10)
            _, ax = plt.subplots()
            ax.bar(range(len(most_common_words)), [count for word, count in most_common_words])
            ax.set_xticks(range(len(most_common_words)))
            ax.set_xticklabels([word for word, count in most_common_words])
            plt.show()
            return ax, most_common_words
    except FileNotFoundError:
        raise FileNotFoundError("CSV file not found.")
    except IOError:
        raise IOError("Error reading file.")