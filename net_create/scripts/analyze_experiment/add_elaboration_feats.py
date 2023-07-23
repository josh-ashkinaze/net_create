



import pandas as pd
import numpy as np
import spacy
import os
import requests
import bz2
from spacy.lang.en.stop_words import STOP_WORDS
from joblib import Parallel, delayed


def get_tfidf_file():
    url = 'https://www.ideals.illinois.edu/items/91826/bitstreams/285420/object?dl=1'
    fn = '../../data/experiment_data/tfidf_weights.csv'
    if not os.path.exists(fn):
        response = requests.get(url)
        response.raise_for_status()
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        bz2_path = fn + '.bz2'
        with open(bz2_path, 'wb') as f:
            f.write(response.content)
        with bz2.BZ2File(bz2_path, 'rb') as f_in, open(fn, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(bz2_path)

def process_text(tfidf_filtered, tokens, stopwords):
    non_stopword_count = sum(1 for token in tokens if token not in stopwords)
    idf_scores_ibf = tfidf_filtered[tfidf_filtered['token'].isin(tokens)]['IBF'].values
    sum_ibf = np.sum(idf_scores_ibf)
    idf_scores_ipf = tfidf_filtered[tfidf_filtered['token'].isin(tokens)]['IPF'].values
    sum_ipf = np.sum(idf_scores_ipf)
    return sum_ibf, sum_ipf, non_stopword_count


if __name__ == "__main__":
    get_tfidf_file()
    tfidf = pd.read_csv("../../data/experiment_data/tfidf_weights.csv")
    df = pd.read_csv("../../data/experiment_data/data_clean.csv")
    df = df.dropna(subset=['response_text'])

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

    # get all unique tokens in one go
    all_docs = list(nlp.pipe(df['response_text'], batch_size=500))
    all_tokens = {token.text for doc in all_docs for token in doc if token.is_alpha}

    # filter tfidf once for all tokens
    tfidf_filtered = tfidf[tfidf['token'].isin(all_tokens)]

    results = Parallel(n_jobs=-1)(
        delayed(process_text)(tfidf_filtered, [token.text for token in doc if token.is_alpha], STOP_WORDS) for doc in
        all_docs)

    df['elab_ibf'], df['elab_ipf'], df['elab_sw'] = zip(*results)
    df['elab_wc'] = df['response_text'].apply(lambda x: len(x.split()))
    elab_df = df[['response_id', 'elab_ibf', 'elab_ipf', 'elab_sw', 'elab_wc']]
    elab_df.to_csv("../../data/experiment_data/expr_data_elab.csv")