import unicodedata
import csv
from collections import Counter
import matplotlib.pyplot as plt

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def word_count_from_csv(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        all_words = []
        for row in reader:
            for word in row:
                normalized_word = normalize_text(word)
                all_words.append(normalized_word)
    return all_words

def plot_word_frequencies(word_counts):
    top_words = word_counts.most_common(10)
    words, frequencies = zip(*top_words)

    fig, ax = plt.subplots()
    ax.bar(words, frequencies)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequencies')
    ax.set_title('Top 10 Most Common Words')
    plt.xticks(rotation=45)
    return ax, list(top_words)

def task_func(csv_file):
    all_words = word_count_from_csv(csv_file)
    word_counts = Counter(all_words)
    ax, most_common_words = plot_word_frequencies(word_counts)
    return ax, most_common_words

# Example usage
# create_dummy_csv_file('dummy.csv')
# ax, most_common_words = task_func('dummy.csv')
# os.remove('dummy.csv')
# print(type(ax))
# print(type(most_common_words))