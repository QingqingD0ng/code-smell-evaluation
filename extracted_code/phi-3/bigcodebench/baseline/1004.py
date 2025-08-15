import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(url):
    response = urllib.request.urlopen(url)
    html = response.read().decode('utf-8')
    text = re.findall(r'\b\w+\b', html)
    word_freq = Counter(text)
    ax = plt.figure().gca()
    top_words = word_freq.most_common(10)
    plt.bar([word for word, count in top_words], [count for word, count in top_words])
    plt.show()
    return word_freq, ax