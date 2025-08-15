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
    plt.bar(words, counts)
    ax = plt.gca()
    plt.show()
    
    return common_words, ax