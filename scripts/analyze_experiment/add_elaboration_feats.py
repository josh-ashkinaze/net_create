"""
Author: Joshua Ashkinaze
Date: 2023-07-11

Description: This script contains functions for adding elaboration to AUT data

"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed
import logging
import os
import sys
import requests
import bz2

LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

class StreamToLogger(object):
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level

    def write(self, message):
       if message.rstrip() != "":
          self.logger.log(self.level, message.rstrip())

    def flush(self):
       pass

stdout_logger = logging.getLogger('STDOUT')
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

stderr_logger = logging.getLogger('STDERR')
sl = StreamToLogger(stderr_logger, logging.ERROR)
sys.stderr = sl


def get_tfidf_file():
    url = 'https://www.ideals.illinois.edu/items/91826/bitstreams/285420/object?dl=1'
    fn = '../../data/experiment_data/tfidf_weights.csv'

    # Check if the file exists
    if not os.path.exists(fn):
        # Download the file
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while downloading file: {e}")
            return

        # Create directories if they don't exist
        try:
            os.makedirs(os.path.dirname(fn), exist_ok=True)
        except Exception as e:
            print(f"Error occurred while creating directories: {e}")
            return

        bz2_path = fn + '.bz2'
        try:
            with open(bz2_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error occurred while writing bz2 file: {e}")
            return

        # Decompress the file
        try:
            with bz2.BZ2File(bz2_path, 'rb') as f_in, open(fn, 'wb') as f_out:
                f_out.write(f_in.read())
        except Exception as e:
            print(f"Error occurred while decompressing bz2 file: {e}")
            return

        # Remove the compressed file
        try:
            os.remove(bz2_path)
        except Exception as e:
            print(f"Error occurred while removing bz2 file: {e}")
            return
    else:
        print(f'File {fn} already exists.')

def make_stopwords():
    # https://raw.githubusercontent.com/explosion/spaCy/a741de7cf658ce9a90d7afe67c88face8fb658ad/spacy/lang/en/stop_words.py
    STOP_WORDS = set(
        """
    a about above across after afterwards again against all almost alone along
    already also although always am among amongst amount an and another any anyhow
    anyone anything anyway anywhere are around as at

    back be became because become becomes becoming been before beforehand behind
    being below beside besides between beyond both bottom but by

    call can cannot ca could

    did do does doing done down due during

    each eight either eleven else elsewhere empty enough even ever every
    everyone everything everywhere except

    few fifteen fifty first five for former formerly forty four from front full
    further

    get give go

    had has have he hence her here hereafter hereby herein hereupon hers herself
    him himself his how however hundred

    i if in indeed into is it its itself

    keep

    last latter latterly least less

    just

    made make many may me meanwhile might mine more moreover most mostly move much
    must my myself

    name namely neither never nevertheless next nine no nobody none noone nor not
    nothing now nowhere

    of off often on once one only onto or other others otherwise our ours ourselves
    out over own

    part per perhaps please put

    quite

    rather re really regarding

    same say see seem seemed seeming seems serious several she should show side
    since six sixty so some somehow someone something sometime sometimes somewhere
    still such

    take ten than that the their them themselves then thence there thereafter
    thereby therefore therein thereupon these they third this those though three
    through throughout thru thus to together too top toward towards twelve twenty
    two

    under until up unless upon us used using

    various very very via was we well were what whatever when whence whenever where
    whereafter whereas whereby wherein whereupon wherever whether which while
    whither who whoever whole whom whose why will with within without would

    yet you your yours yourself yourselves
    """.split()
    )

    contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
    STOP_WORDS.update(contractions)

    for apostrophe in ["‘", "’"]:
        for stopword in contractions:
            STOP_WORDS.add(stopword.replace("'", apostrophe))
    return STOP_WORDS

def process_sentence(tfidf, sentence, stopwords):
    tokens = word_tokenize(sentence.lower())
    try:
        non_stopword_count = sum(1 for token in tokens if token not in stopwords)
    except Exception as e:
        logging.info(f"Error in processing sentence stopwords: {sentence}. Error: {str(e)}")
        non_stopword_count = np.NaN
    try:
        idf_scores = tfidf[tfidf['token'].isin(tokens)]['IBF'].values
        sum_ibf = np.sum(idf_scores)
    except Exception as e:
        logging.info(f"Error in processing sentence idf scores: {sentence}. Error: {str(e)}")
        sum_ibf = np.NaN

    try:
        idf_scores = tfidf[tfidf['token'].isin(tokens)]['IPF'].values
        sum_ipf = np.sum(idf_scores)
    except Exception as e:
        logging.info(f"Error in processing sentence idf scores: {sentence}. Error: {str(e)}")
        sum_ipf = np.NaN
    return sum_ibf, sum_ipf, non_stopword_count

if __name__ == "__main__":


    get_tfidf_file()
    tfidf = pd.read_csv("../../data/experiment_data/tfidf_weights.csv")
    stop_words = make_stopwords()
    df = pd.read_csv("../../data/experiment_data/expr_data.csv")
    df = df.dropna(subset=['response_text'])

    # Process each sentence in parallel
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_sentence)(tfidf, sentence, stop_words) for sentence in df['response_text'])

    # Unpack results into two new columns
    df['elab_ibf'], df['elab_ipf'], df['elab_sw'] = zip(*results)
    df['elab_wc'] = df['response_text'].apply(lambda x: len(x.split()))
    elab_df = df[['response_id', 'elab_ibf', 'elab_ipf', 'elab_sw', 'elab_wc']]
    elab_df.to_csv("../../data/experiment_data/expr_data_elab.csv")

