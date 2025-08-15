import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    word_counts = Counter(words)
    most_common = word_counts.most_common(10)
    
    labels, values = zip(*most_common)
    
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return most_common, ax