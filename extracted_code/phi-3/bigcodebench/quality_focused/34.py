from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt

def task_func(text):
    url_regex = r'https?://\S+'
    text_no_urls = re.sub(url_regex, '', text)
    words = re.findall(r'\w+', text_no_urls)
    if not words:
        raise ValueError("No words available to generate a word cloud after removing URLs.")
    word_frequencies = {word: words.count(word) for word in set(words)}
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_frequencies)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return wordcloud