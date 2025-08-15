import re

import nltk

from sklearn.decomposition import NMF

from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')


ALPHANUMERIC = re.compile('[\W_]+')

STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def task_func(texts, num_topics):

    texts = [ALPHANUMERIC.sub(' ', text.lower()) for text in texts]

    texts = [' '.join([word for word in text.split() if word not in STOPWORDS]) for text in texts]

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(texts)

    nmf = NMF(n_components=num_topics, random_state=42).fit(tfidf)

    feature_names = vectorizer.get_feature_names_out()

    topics = []

    for topic_idx, topic in enumerate(nmf.components_):

        topic_words = [feature_names[i] for i in topic.argsort()[:-6 - 1:-1]]

        topics.append(topic_words)

    return topics