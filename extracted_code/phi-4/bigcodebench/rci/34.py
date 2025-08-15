def task_func(text):
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    
    text = re.sub(r'https?://\S+', '', text)
    words = ''.join(re.findall(r'\b\w+\b', text))
    
    if not words:
        raise ValueError("No words available to generate a word cloud after removing URLs.")
    
    try:
        wordcloud = WordCloud(width=800, height=600).generate(words)
    except Exception as e:
        raise RuntimeError("Failed to generate word cloud.") from e
    
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    return wordcloud.words_