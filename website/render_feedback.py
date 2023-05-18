import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore
from helpers import helpers as my_utils
import base64
import spacy
import io
from scipy.spatial.distance import cosine as cosine_distance

# Load the small english model. You'll need to download it first via spacy's cli: python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_md')


def make_aesthetic():
    sns.set(style='white', context='poster', font_scale=1.1)
    try:
        plt.rcParams.update({'font.family': 'Arial'})
    except:
        pass
    sns.set_palette(sns.color_palette('dark'))
    sns.despine()
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.titlesize'] = 24


def load_data(comparison, prefix="../"):
    if comparison == "human":
        return np.array(pd.read_csv(prefix + "data/prior_responses.csv")['target'].tolist())
    elif comparison == "AI":
        return np.array(pd.read_csv(prefix + "data/scored_ai_responses.csv")['originality'].tolist())


def comparison_graph(participant_scores, comparison, prefix="../"):
    make_aesthetic()
    mean_score = np.mean(participant_scores)
    comparison_scores = load_data(comparison, prefix)

    percentile = int(percentileofscore(comparison_scores, mean_score))

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.histplot(comparison_scores, bins=10, stat='percent', color='#1F4287', alpha=0.8)
    plt.suptitle("Your average response was more creative than\n{}% of {} responses".format(percentile, comparison),
                 ha='left', x=0.125, fontweight='bold', y=1.07, fontsize=26)
    ax.set_title(
        f"The graph shows creativity of {comparison}-generated responses on a scale of 1-5,\nwith a dashed line for your average creativity score.",
        loc='left', pad=10, fontsize=20)
    ax.set_xlabel("Creativity (1-5)")
    ax.set_ylabel(f"Percent of {comparison.title()} Responses")
    ax.axvline(x=mean_score, linewidth=4, color='#FFD23F', linestyle='--')
    plt.savefig("img.png", bbox_inches='tight')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def plot_ai_human(conditions, scores):
    color_dict = {'Human Only': '#1F4287', 'AI + Human': '#FFD23F'}
    df = pd.DataFrame({'conditions': conditions, 'scores': scores})
    df['source'] = df['conditions'].apply(lambda x: "Human Only" if x == 'h' else 'AI + Human')
    means = df.groupby(by='source')['scores'].mean().reset_index()
    no_ai = means.query("source=='Human Only'")['scores'].tolist()[0]
    with_ai = means.query("source=='AI + Human'")['scores'].tolist()[0]

    if with_ai >= no_ai:
        adj = "more"
    else:
        adj = "less"
    percent_diff = abs(int(((with_ai - no_ai) / (no_ai)) * 100))
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.suptitle(
        f"Your responses were {percent_diff}% {adj} creative when\nexposed to AI ideas than when exposed\nto only human ideas.",
        ha='left', x=0.125, fontweight='bold', y=1.07, fontsize=26)
    sns.barplot(data=means, x='source', y='scores', ci=False, palette=color_dict)
    ax.set_xlabel("Idea Exposure")
    ax.set_ylabel("Average Creativity (1-5)")
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight')
    buf2.seek(0)
    img_base64_2 = base64.b64encode(buf2.read()).decode('utf-8')
    return img_base64_2


def make_graphs(participant_responses, conditions, file_prefix="../"):
    participant_scores = my_utils.batch_score_responses(participant_responses)['originality'].tolist()
    ai_graph = comparison_graph(participant_scores, "AI", file_prefix)
    human_graph = comparison_graph(participant_scores, "human", file_prefix)
    ai_human_graph = plot_ai_human(conditions, participant_scores)
    return human_graph, ai_graph, ai_human_graph


def calculate_similarity(sentence1, sentence2):
    # Transform sentences into their vector representation
    sentence1_vector = nlp(sentence1).vector
    sentence2_vector = nlp(sentence2).vector

    # Reshape vectors to 1-D numpy arrays
    sentence1_vector = sentence1_vector.ravel()
    sentence2_vector = sentence2_vector.ravel()

    return 1 - cosine_distance(sentence1_vector, sentence2_vector)
