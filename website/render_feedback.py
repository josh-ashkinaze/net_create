import base64
import io
import random
from functools import lru_cache

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec as w2v
from gensim.utils import simple_preprocess as preprocess
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import percentileofscore

from helpers import helpers as my_utils

model = w2v.load("data/filtered_w2v.wv")


def make_aesthetic():
    sns.set(style='white', context='poster', font_scale=1.1)
    try:
        plt.rcParams.update({'font.family': 'Roboto'})
    except:
        pass
    sns.set_palette(sns.color_palette('dark'))
    sns.despine()
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.titlesize'] = 24

@lru_cache(maxsize=128)
def load_data(comparison, prefix="../"):
    if comparison == "human":
        return np.array(pd.read_csv(prefix + "data/scored_human_prior_responses.csv")['predicted'].tolist())
    elif comparison == "AI":
        return np.array(pd.read_csv(prefix + "data/scored_ai_responses.csv")['originality'].tolist())


def comparison_graph(participant_scores, comparison, prefix="../"):
    make_aesthetic()
    mean_score = np.mean(participant_scores)
    comparison_scores = load_data(comparison, prefix)

    percentile = int(percentileofscore(comparison_scores, mean_score))

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.histplot(comparison_scores, bins=10, stat='percent', color='#1F4287', alpha=0.8)
    plt.suptitle("Your average response was more creative than\n{}% of {} responses.".format(percentile, comparison),
                 ha='left', x=0.125, fontweight='bold', y=1.07, fontsize=26)
    ax.set_title(
        f"The graph shows creativity of {comparison}-generated responses on a scale of 1-5,\nwith a dashed line for your average creativity score.",
        loc='left', pad=10, fontsize=20)
    ax.set_xlabel("Creativity (1-5)")
    ax.set_ylabel(f"Percent of {comparison.title()} Responses")
    ax.axvline(x=mean_score, linewidth=4, color='#E1DD8F', linestyle='--')
    plt.savefig("img.png", bbox_inches='tight')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def plot_ai_human(conditions, scores):
    make_aesthetic()
    color_dict = {'Human Only': '#1F4287', 'AI + Human': '#E1DD8F'}
    df = pd.DataFrame({'conditions': conditions, 'scores': scores})
    df['source'] = df['conditions'].apply(lambda x: "Human Only" if x == 'h' else 'AI + Human')
    means = df.groupby(by='source')['scores'].mean().reset_index()

    # Vectorized operations
    no_ai, with_ai = means.set_index('source').loc[['Human Only', 'AI + Human'], 'scores']
    adj = "more" if with_ai >= no_ai else "less"
    percent_diff = abs(int(((with_ai - no_ai) / no_ai) * 100))
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.suptitle(
        f"Your responses were {percent_diff}% {adj} creative when\nexposed to AI ideas than when exposed\nto only human ideas.",
        ha='left', x=0.125, fontweight='bold', y=1.07, fontsize=26)
    sns.barplot(data=means, x='source', y='scores', errorbar=('ci', False), palette=color_dict)
    ax.set_xlabel("Idea Exposure")
    ax.set_ylabel("Average Creativity (1-5)")
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight')
    buf2.seek(0)
    img_base64_2 = base64.b64encode(buf2.read()).decode('utf-8')
    return img_base64_2


def make_graphs(participant_responses, conditions, file_prefix="../", participant_scores=None):
    """
    Generates graphs for the feedback page.

    If we are doing this from UUID then we can pass in participant_scores directly.
    Otherwise, we need to score the responses first.

    :param participant_responses: A zip file of the participant's responses for each item
    :param conditions: A list of item conditions
    :param file_prefix: Where to locate comparison files
    :param participant_scores: Optional, include if we are doing this from UUID because already scored
    :return: graphs for human, AI, and AI + human comparisons
    """
    font_path = file_prefix + "website/resources/Roboto.ttf"
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    if not participant_scores:
        participant_scores = my_utils.score_aut_responses(participant_responses)['originality'].tolist()
    else:
        pass
    ai_graph = comparison_graph(participant_scores, "AI", file_prefix)
    human_graph = comparison_graph(participant_scores, "human", file_prefix)
    ai_human_graph = plot_ai_human(conditions, participant_scores)
    return human_graph, ai_graph, ai_human_graph, participant_scores


def sentence_vector(sentence, model):
    words = preprocess(sentence)
    word_vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if not word_vectors:
        return None
    return np.mean(word_vectors, axis=0)


def calculate_similarity(sentence1, sentence2):
    try:
        # Remove labels
        sentence1 = sentence1.split('<span', 1)[0].strip()
        sentence2 = sentence2.split('<span', 1)[0].strip()
        sentence1_vector = sentence_vector(sentence1, model)
        sentence2_vector = sentence_vector(sentence2, model)
        if sentence1_vector is None or sentence2_vector is None:
            print("Get a NONE", sentence1, sentence2)
            return random.uniform(0.3, 0.7)

        similarity = 1 - cosine_distance(sentence1_vector, sentence2_vector)
        print("Similarity loaded successfully")
        return similarity
    except Exception as e:
        print("ERROR", sentence1, sentence2, e)
        return random.uniform(0.3, 0.7)
