import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(10)
    words, counts = zip(*common_words)
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Most Common Words')
    return common_words, ax