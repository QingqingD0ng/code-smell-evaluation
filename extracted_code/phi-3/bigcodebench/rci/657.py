import re

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from gensim.models import Word2Vec


ALPHANUMERIC = re.compile(r'[\W_]+')


def clean_text(text):

    text = ALPHANUMERIC.sub(' ', text)

    text = text.lower()

    tokens = word_tokenize(text)

    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

    return filtered_tokens


def train_word2vec_model(texts, vector_size=100, window=5, min_count=1, workers=4):

    cleaned_texts = [clean_text(text) for text in texts]

    model = Word2Vec(sentences=cleaned_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    return model


def main():

    texts = ["Hello, World!", "Machine Learning is great", "Python is my favorite programming language"]

    model = train_word2vec_model(texts)

    vector = model.wv['python']


if __name__ == "__main__":

    main()