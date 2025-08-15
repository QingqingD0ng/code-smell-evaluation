import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]


def task_func(file_path, save_path=None):
    try:
        # Read CSV file into a DataFrame
        df = pd.read_csv(file_path)
        if df.shape[1] == 0:  # Check if there's no header
            df = pd.read_csv(file_path, header=None, names=['Text'])

        # Text data extraction
        text_data = df['Text'].dropna().tolist()

        # Vectorize text data and filter out stopwords
        vectorizer = CountVectorizer(stop_words=STOP_WORDS)
        word_counts = vectorizer.fit_transform(text_data)
        words = vectorizer.get_feature_names_out()
        frequencies = word_counts.toarray().sum(axis=0)
        freq_dict = dict(zip(words, frequencies))

        # Sort the dictionary by frequency and get the top 10 words
        top_words = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        top_words_list = [word for word, freq in top_words]

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.bar(top_words_list, [freq for word, freq in top_words])
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Common Words')

        # Save or show the plot based on save_path
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return plt.gca() if not save_path else None

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return