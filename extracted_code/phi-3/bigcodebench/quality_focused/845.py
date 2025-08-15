import re
import numpy as np
from collections import Counter
from Levenshtein import ratio

ALPHANUMERIC = re.compile('[\W_]+')

def clean_and_tokenize(text):
    return re.sub(ALPHANUMERIC,'', text.lower()).split()

def term_frequency_matrix(text):
    tokens = clean_and_tokenize(text)
    total_tokens = len(tokens)
    freq_vector = Counter(tokens)
    tf_matrix = np.zeros((total_tokens, 1))
    for i, token in enumerate(tokens):
        tf_matrix[i] = freq_vector[token] / total_tokens
    return tf_matrix

def cosine_similarity(matrix1, matrix2):
    dot_product = np.dot(matrix1.T, matrix2)
    norm_matrix1 = np.linalg.norm(matrix1)
    norm_matrix2 = np.linalg.norm(matrix2)
    return dot_product / (norm_matrix1 * norm_matrix2) if norm_matrix1 * norm_matrix2 else 0

def task_func(text1, text2):
    tf_matrix1 = term_frequency_matrix(text1)
    tf_matrix2 = term_frequency_matrix(text2)
    cosine_sim = cosine_similarity(tf_matrix1, tf_matrix2)
    levenshtein_ratio = ratio(text1, text2)
    return cosine_sim, levenshtein_ratio