import re
import numpy as np
from collections import Counter
from Levenshtein import ratio

ALPHANUMERIC = re.compile('[\W_]+')

def task_func(text1, text2):
    text1 = ALPHANUMERIC.sub(' ', text1).lower()
    text2 = ALPHANUMERIC.sub(' ', text2).lower()

    terms1 = Counter(text1.split())
    terms2 = Counter(text2.split())

    all_terms = set(terms1).union(set(terms2))
    term_vectors = [
        np.array([terms1.get(term, 0) for term in all_terms], dtype=int),
        np.array([terms2.get(term, 0) for term in all_terms], dtype=int)
    ]

    cosine_sim = np.dot(term_vectors[0], term_vectors[1]) / (np.linalg.norm(term_vectors[0]) * np.linalg.norm(term_vectors[1]))
    levenshtein_ratio = ratio(text1, text2)
    
    return (cosine_sim, levenshtein_ratio)