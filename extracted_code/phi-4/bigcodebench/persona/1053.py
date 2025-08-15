import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Constants
STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]

def task_func(file_path, save_path=None):
    try:
        # Read CSV file
        df = pd.read_csv(file_path, header=None if 'Text' not in df.columns else 0)
        text_data = df['Text'] if 'Text' in df.columns else df[0]

        # Initialize CountVectorizer with custom stop words
        vectorizer = CountVectorizer(stop_words=STOP_WORDS)
        word_counts = vectorizer.fit_transform(text_data)

        # Sum word counts and get feature names
        sum_words = word_counts.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # Get the top 10 most common words
        top_words = words_freq[:10]
        words, counts = zip(*top_words)

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.bar(words, counts, color='skyblue')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Common Words')
        plt.xticks(rotation=45)

        # Save or display plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return None
        else:
            return plt.gca()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None