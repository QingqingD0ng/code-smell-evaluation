import re
import numpy as np
from collections import Counter
from Levenshtein import ratio

ALPHANUMERIC = re.compile('[\W_]+')

def clean_text(text):
    return ALPHANUMERIC.sub(' ', text).lower()

def term_frequency(text):
    return Counter(text.split())

def cosine_similarity(freq1, freq2):
    terms = set(freq1) | set(freq2)
    vec1 = np.array([freq1.get(term, 0) for term in terms])
    vec2 = np.array([freq2.get(term, 0) for term in terms])
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

def task_func(text1, text2):
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    freq1 = term_frequency(clean1)
    freq2 = term_frequency(clean2)
    
    cosine_sim = cosine_similarity(freq1, freq2)
    levenshtein_ratio = ratio(clean1, clean2)
    
    return cosine_sim, levenshtein_ratio