import re
import nltk
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure nltk's stopwords are downloaded
nltk.download('stopwords')

# Constants
ALPHANUMERIC = re.compile('[\W_]+')
STOPWORDS = nltk.corpus.stopwords.words('english')

def preprocess_text(text):
    text = ALPHANUMERIC.sub(' ', text).lower()
    return ''.join(word for word in text.split() if word not in STOPWORDS)

def task_func(texts, num_topics):
    processed_texts = [preprocess_text(text) for text in texts]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_model.fit(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-11:-1]])
    return topics