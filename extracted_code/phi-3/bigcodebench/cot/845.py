import re
import numpy as np
from collections import Counter
from Levenshtein import ratio

ALPHANUMERIC = re.compile('[\W_]+')

def clean_text(text):
    return ALPHANUMERIC.sub(' ', text).lower()

def term_frequency(text):
    words = text.split()
    word_count = Counter(words)
    total_words = sum(word_count.values())
    return {word: freq / total_words for word, freq in word_count.items()}

def cosine_similarity(tf1, tf2):
    words = set(tf1.keys()) | set(tf2.keys())
    dot_product = sum(tf1.get(word, 0) * tf2.get(word, 0) for word in words)
    norm1 = np.sqrt(sum(val ** 2 for val in tf1.values()))
    norm2 = np.sqrt(sum(val ** 2 for val in tf2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

def task_func(text1, text2):
    clean_text1, clean_text2 = clean_text(text1), clean_text(text2)
    tf1, tf2 = term_frequency(clean_text1), term_frequency(clean_text2)
    cosine_sim = cosine_similarity(tf1, tf2)
    levenshtein_ratio = ratio(clean_text1, clean_text2)
    return cosine_sim, levenshtein_ratio