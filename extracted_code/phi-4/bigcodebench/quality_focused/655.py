import re
import nltk
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
ALPHANUMERIC = re.compile('[\W_]+')
STOPWORDS = nltk.corpus.stopwords.words('english')

def task_func(texts, num_topics):
    def preprocess(text):
        text = ALPHANUMERIC.sub(' ', text).lower()
        return''.join([word for word in text.split() if word not in STOPWORDS])

    processed_texts = [preprocess(text) for text in texts]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    nmf_model = NMF(n_components=num_topics, random_state=0)
    nmf_model.fit(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-6:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics.append(top_features)
    
    return topics