import pandas as pd
import numpy as np
import re
import sys
from gensim.models import KeyedVectors
# Some useful libraries
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer


# --- Data Loading ---
def load_data():
    df = pd.read_csv("data/aligned_literature_en_es.csv")
    en_model = KeyedVectors.load_word2vec_format("data/mini.en.vec")
    es_model = KeyedVectors.load_word2vec_format("data/mini.es.vec")
    return df, en_model, es_model


# --- Evaluation ---
def run_eval(en_vecs, es_vecs, label, metric='cosine'):
    print(f"\n--- {label} Results ---")
    for dir_name, q, g in [("EN->ES", en_vecs, es_vecs), ("ES->EN", es_vecs, en_vecs)]:
        acc1 = compute_accuracy(q, g, metric, k=1)
        acc3 = compute_accuracy(q, g, metric, k=3)
        print(f"{dir_name} | Top-1: {acc1:.2%} | Top-3: {acc3:.2%}")

        
def compute_accuracy(query_vecs, gallery_vecs, metric='cosine', k=1):
    # ----------------------
    # REPLACE WITH YOUR CODE
    return 0
    # ----------------------

    
def main():
    df, en_model, es_model = load_data()   
    
    # 1. Baseline (Simple Mean)
    # ----------------------
    # REPLACE WITH YOUR CODE
    en_base = np.zeros((len(df), en_model.vector_size))
    es_base = np.zeros((len(df), es_model.vector_size)) 
    # ----------------------
    run_eval(en_base, es_base, "Baseline (Simple Mean)")
    
    # 2. TF-IDF Weighted (No Centering)
    # ----------------------
    # REPLACE WITH YOUR CODE
    en_tfidf_vecs = np.zeros((len(df), en_model.vector_size))
    es_tfidf_vecs = np.zeros((len(df), es_model.vector_size))
    # ----------------------
    run_eval(en_tfidf_vecs, es_tfidf_vecs, "TF-IDF Weighted")

    
    # Global means for Centering
    # ----------------------
    # REPLACE WITH YOUR CODE
    mean_en = np.zeros(en_model.vector_size)
    mean_es = np.zeros(es_model.vector_size)
    mean_en_tfidf = np.zeros(es_model.vector_size)
    mean_es_tfidf = np.zeros(es_model.vector_size)
    # ----------------------
    
    # 3. Mean-Centered (No TF-IDF) 
    run_eval(en_base - mean_en, es_base - mean_es, "Mean-Centered (Simple)")
    
    # 4. Mean-Centered + TF-IDF
    run_eval(en_tfidf_vecs - mean_en_tfidf, es_tfidf_vecs - mean_es_tfidf, "Mean-Centered + TF-IDF")

if __name__ == "__main__":
    main()
