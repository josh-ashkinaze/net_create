import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore
from helpers import helpers as my_utils
import base64
import io

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
    prefix = "../"
    if comparison == "human":
        return np.array(pd.read_csv(prefix + "data/prior_responses.csv")['target'].tolist())
    elif comparison == "AI":
        return np.array(pd.read_csv(prefix + "data/scored_ai_responses.csv")['originality'].tolist())

def graph_score(participant_responses, comparison, prefix="../"):
    participant_scores= my_utils.batch_score_responses(participant_responses)['originality'].tolist()
    make_aesthetic()
    mean_score = np.mean(participant_scores)
    comparison_scores = load_data(comparison, prefix)

    percentile = int(percentileofscore(comparison_scores, mean_score))

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.histplot(comparison_scores, bins=10, stat='percent', color='#1F4287', alpha=0.8)
    plt.suptitle("Your average response was more creative than\n{}% of {} responses".format(percentile, comparison), ha='left', x=0.125, fontweight='bold',y=1.07, fontsize=26)
    ax.set_title(f"The graph shows creativity of {comparison}-generated responses on a scale of 1-5,\nwith a dashed line for your average creativity score.", loc='left', pad=10, fontsize=20)
    ax.set_xlabel("Creativity (1-5)")
    ax.set_ylabel(f"Percent of {comparison.title()} Responses")
    ax.axvline(x=mean_score, linewidth=4, color='#FFD23F', linestyle='--')
    plt.savefig("img.png", bbox_inches='tight')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64



