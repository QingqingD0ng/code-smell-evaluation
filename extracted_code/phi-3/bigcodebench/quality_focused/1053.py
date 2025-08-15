import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]

def task_func(file_path, save_path=None):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        if df.shape[1] == 1:
            text_data = df.iloc[:, 0]
        else:
            text_data = df['Text']

        # Create a CountVectorizer instance
        vectorizer = CountVectorizer(stop_words=STOP_WORDS)
        # Fit and transform the data
        word_counts = vectorizer.fit_transform(text_data)
        # Sum the word counts to get the most common words
        sums = word_counts.sum(axis=0)
        words_freq = [(word, sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:10]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar([x[0] for x in words_freq], [x[1] for x in words_freq])
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Common Words')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            return plt.gca()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except Exception as e:
        print(e)
        return None