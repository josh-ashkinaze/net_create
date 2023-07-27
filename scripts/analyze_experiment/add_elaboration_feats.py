import pandas as pd
import numpy as np
import spacy
import os
import requests
import logging
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
    n_tokens = len(tokens)
    non_stopword_count = sum(1 for token in tokens if token not in stopwords)
    idf_scores_ibf = tfidf_filtered[tfidf_filtered['token'].isin(tokens)]['IBF'].values
    sum_ibf = np.sum(idf_scores_ibf)
    idf_scores_ipf = tfidf_filtered[tfidf_filtered['token'].isin(tokens)]['IPF'].values
    sum_ipf = np.sum(idf_scores_ipf)
    return sum_ibf, sum_ipf, non_stopword_count, n_tokens


if __name__ == "__main__":
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
    get_tfidf_file()
    tfidf = pd.read_csv("../../data/experiment_data/tfidf_weights.csv")
    df = pd.read_csv("../../data/experiment_data/data_clean.csv")
    logging.info("length of df before dropping na: " + str(len(df)))
    df = df.dropna(subset=['response_text'])
    logging.info("length of df after dropping na: " + str(len(df)))

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

    # get all unique tokens in one go
    all_docs = list(nlp.pipe(df['response_text'], batch_size=500))
    all_tokens = {token.text for doc in all_docs for token in doc if token.is_alpha}
    logging.info("All tokens extracted")

    # filter tfidf once for all tokens
    tfidf_filtered = tfidf[tfidf['token'].isin(all_tokens)]
    logging.info("TFIDF filtered")

    results = Parallel(n_jobs=-1)(
        delayed(process_text)(tfidf_filtered, [token.text for token in doc if token.is_alpha], STOP_WORDS) for doc in
        all_docs)


    df['elab_ibf'], df['elab_ipf'], df['elab_not_sw'], df['elab_n_tokens'] = zip(*results)
    logging.info("length of df" + str(len(df)))
    logging.info("Elaboration features added")
    df.to_csv("../../data/experiment_data/data_clean_with_elab.csv", index=False)
    logging.info("Data saved")
