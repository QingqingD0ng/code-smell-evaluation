import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def task_func(text):
    text = re.sub(r'https?://\S+', '', text)
    words =''.join(re.findall(r'\b\w+\b', text))
    if not words:
        raise ValueError("No words available to generate a word cloud after removing URLs.")
    wordcloud = WordCloud().generate(words)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return wordcloud