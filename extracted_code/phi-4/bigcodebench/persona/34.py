import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def task_func(text):
    # Remove URLs from the text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Check if wordcloud has any words
    if not wordcloud.words_:
        raise ValueError("No words available to generate a word cloud after removing URLs.")
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    return wordcloud