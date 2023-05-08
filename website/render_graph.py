import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore

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

def load_data(comparison):
    prefix = "../"
    if comparison == "human":
        return np.array(pd.read_csv("../data/prior_responses.csv")['target'].tolist())
    elif comparison == "AI":
        return np.array(pd.read_csv("../data/scored_ai_responses.csv")['originality'].tolist())

def graph_score(participant_scores, comparison):
    make_aesthetic()
    mean_score = np.mean(participant_scores)
    comparison_scores = load_data(comparison)

    percentile = int(percentileofscore(comparison_scores, mean_score))

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.histplot(comparison_scores, bins=10, stat='percent', color='#5E2BFF', alpha=0.8)
    plt.suptitle("Your average response was more creative than\n{}% of {} responses".format(percentile, comparison), ha='left', x=0.125, fontweight='bold',y=1.07, fontsize=26)
    ax.set_title(f"The graph shows creativity of {comparison}-generated responses on a scale of 1-5,\nwith a red line for your average creativity score.", loc='left', pad=10, fontsize=20)
    ax.set_xlabel("Creativity (1-5)")
    ax.set_ylabel(f"Percent of {comparison.title()} Responses")
    ax.axvline(x=mean_score, linewidth=4, color='#D7263D', linestyle='--')
    return ax


