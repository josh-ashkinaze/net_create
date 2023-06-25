"""
Author: Joshua Ashkinaze
Date: 2023-06-25

Description: Filter top 15k w2v words
"""


import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
def filter_top_words(model, top_k):
    """
    Filters the model to only include vectors for the top_k words.

    Returns:
    gensim.models.Word2Vec: A Word2Vec model containing vectors only for the top_k words.
    """
    filtered_model = Word2Vec(vector_size=model.vector_size)
    filtered_model.wv = gensim.models.keyedvectors.KeyedVectors(vector_size=model.vector_size)
    for i, word in enumerate(model.index_to_key[:top_k]):
        filtered_model.wv.add_vector(word, model[word])
        filtered_model.wv.set_vecattr(word, 'count', model.get_vecattr(word, 'count'))
    return filtered_model


def main():
    # Download the pre-trained word2vec model
    w2v_model = api.load("word2vec-google-news-300")
    top_k = 15000
    filtered_w2v = filter_top_words(w2v_model, top_k)
    filename = '../../data/filtered_w2v.wv'
    filtered_w2v.save(filename)

if __name__ == "__main__":
    main()
