import re
import numpy as np
from collections import Counter
from Levenshtein import ratio

ALPHANUMERIC = re.compile('[\W_]+')

def task_func(text1, text2):
    clean_text1 = ALPHANUMERIC.sub(' ', text1).lower()
    clean_text2 = ALPHANUMERIC.sub(' ', text2).lower()
    
    counter1 = Counter(clean_text1.split())
    counter2 = Counter(clean_text2.split())
    
    all_terms = set(counter1.keys()).union(set(counter2.keys()))
    
    vec1 = np.array([counter1.get(term, 0) for term in all_terms])
    vec2 = np.array([counter2.get(term, 0) for term in all_terms])
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    cosine_similarity = dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0
    
    levenshtein_ratio = ratio(clean_text1, clean_text2)
    
    return cosine_similarity, levenshtein_ratio