import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(url):
    response = urllib.request.urlopen(url)
    text = response.read().decode('utf-8')

    words = re.findall(r'\b\w+\b', text)
    word_counts = Counter(words)

    most_common_words = word_counts.most_common(10)

    plt.bar([word for word, count in most_common_words], [count for word, count in most_common_words])
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Frequent Words')
    plt.show()

    return type('CounterWrapper', (object,), {'most_common': lambda self, n: [(word, count) for word, count in self.most_common(n)]}), plt

# Example usage:
# Assuming the provided example URL points to a valid text file
word_freq, ax = task_func('http://www.example.com/data.txt')
print(word_freq.most_common(5))