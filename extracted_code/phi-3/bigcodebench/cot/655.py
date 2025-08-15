import re
import nltk
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

ALPHANUMERIC = re.compile('[\W_]+')
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(texts):
    processed_texts = []
    for text in texts:
        text = text.lower()
        text = ALPHANUMERIC.sub(' ', text)
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in STOPWORDS]
        processed_texts.append(' '.join(words))
    return processed_texts

def task_func(texts, num_topics):
    processed_texts = preprocess_text(texts)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_model.fit(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-2:-1]]
        topics.append(top_features)
    
    return topics