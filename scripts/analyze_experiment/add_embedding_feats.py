"""
Author: Joshua Ashkinaze
Date: 2023-07-11

Description: This script gets sbert embeddings for sentences

"""

import pandas as pd
import numpy as np
import logging
import os
from sentence_transformers import SentenceTransformer, util

LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                    datefmt='%Y-%m-%d %H:%M:%S', filemode='w')


def main():
    logging.info("Started getting embeddings")
    df = pd.read_csv('../../data/experiment_data/expr_data.csv')
    df = df.dropna(subset=['response_text'])
    df = df[['response_id', 'response_text']]

    ai_ideas = pd.read_csv('../../data/ai_responses.csv')
    ai_ideas = ai_ideas[['response_id', 'response']]
    ai_ideas.columns = ['response_id', 'response_text']

    human_seeds = pd.read_csv('../../data/seed_human_responses.csv')
    human_seeds = human_seeds[['response_id', 'response']]
    human_seeds.columns = ['response_id', 'response_text']

    df = pd.concat([df, ai_ideas, human_seeds])

    model = SentenceTransformer('all-mpnet-base-v2')
    sentences = df['response_text'].tolist()
    embeddings = model.encode(sentences)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    np.save('../../data/experiment_data/cosine_scores.npy', cosine_scores)
    np.save('../../data/experiment_data/embeddings.npy', embeddings)
    rid2idx = {rid: idx for idx, rid in enumerate(df['response_id'])}
    idx2rid = {idx: rid for rid, idx in rid2idx.items()}
    np.save('../../data/experiment_data/rid2idx.npy', rid2idx)
    np.save('../../data/experiment_data/idx2rid.npy', idx2rid)
    logging.info("Saved everything")

if __name__ == "__main__":
    main()
