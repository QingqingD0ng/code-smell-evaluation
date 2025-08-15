import unicodedata
import csv
from collections import Counter
import matplotlib.pyplot as plt

def task_func(csv_file):
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            text_data = []

            for row in reader:
                for cell in row:
                    normalized_text = unicodedata.normalize('NFKD', cell).encode('ascii', 'ignore').decode('ascii')
                    text_data.append(normalized_text.lower())

            words =''.join(text_data).split()
            word_counts = Counter(words)
            most_common_words = word_counts.most_common(10)

            words, frequencies = zip(*most_common_words)
            fig, ax = plt.subplots()
            ax.bar(words, frequencies)
            ax.set_xlabel('Words')
            ax.set_ylabel('Frequency')
            ax.set_title('Top 10 Most Common Words')

            return ax, most_common_words

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {csv_file}")
    except IOError:
        raise IOError(f"Error reading file: {csv_file}")