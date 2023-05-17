import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from helpers.helpers import process_ideas, extract_ideas, find_latest_file


def count_words(x):
    return len(x.split(" "))


def make_aesthetic():
    sns.set(style='white', context='poster', font_scale=0.9)
    plt.rcParams.update({'font.family': 'Arial'})
    sns.set_palette(sns.color_palette('dark'))
    sns.despine()
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.titlesize'] = 22


def make_n_words_graph(mydf):
    plt.figure(figsize=(12, 8))

    custom_colors = ['#7B68EE', '#2F243A', '#2F243A']
    custom_labels = ['Human Responses', 'GPT Responses\n(Prompt = Zero Shot,\nLength Limited)',
                     'GPT Responses\n(Prompt = Zero Shot)']

    avg_n_words = mydf.groupby('prompt_condition')['n_words'].mean().reindex(['prior_work', 'zero_limit', 'zero_shot'])
    sd_n_words = mydf.groupby('prompt_condition')['n_words'].std().reindex(['prior_work', 'zero_limit', 'zero_shot'])
    n_obs = mydf.groupby('prompt_condition')['n_words'].count().reindex(['prior_work', 'zero_limit', 'zero_shot'])

    # Create the barplot
    ax = sns.barplot(data=mydf, x='prompt_condition', y='n_words', order=['prior_work', 'zero_limit', 'zero_shot'],
                     palette=custom_colors)
    for i, (avg, sd, n) in enumerate(zip(avg_n_words, sd_n_words, n_obs)):
        plt.text(i, avg + 1, f'{avg:.2f}', ha='center')
        # plt.text(i, avg+3, f'SD = {sd:.2f}', ha='center')
        # plt.text(i, avg+1, f'n = {n}', ha='center')

    plt.suptitle("Mean Number of Words of AUT Responses", x=0.395, fontweight='bold', y=1)
    plt.title("Error bars are 95% CIs", fontweight='regular', x=-0.043)
    plt.xlabel("")
    plt.ylabel("Number of Words")
    plt.xticks(np.arange(3), custom_labels)  # Replace x-axis labels with your custom labels
    plt.savefig("../../plots/aut_n_words.png", bbox_inches='tight', dpi=300)


def main():
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
    make_aesthetic()
    prior = pd.read_csv("../../data/prior_responses.csv")
    my_items = pd.read_csv("../../data/chosen_aut_items.csv")['aut_item'].tolist()
    prompt_expr = pd.read_json(find_latest_file("../../data/prompt_experiments", "length"), lines=True)
    prompt_expr['prompt_condition'] = prompt_expr["prompt_condition"].replace("zero_shot_limit_length", "zero_limit")
    prompt_expr['ideas'] = prompt_expr['output_responses'].apply(lambda x: extract_ideas(x))
    prompt_exploded = prompt_expr.explode('ideas')
    prompt_exploded['response'] = prompt_exploded['ideas'].apply(lambda x: process_ideas(x))
    prompt_exploded['n_words'] = prompt_exploded['response'].apply(lambda x: count_words(x))
    wdf = prompt_exploded[['aut_item', 'prompt_condition', 'n_words']]
    prior = prior[prior['prompt'].isin(my_items)]
    prior['aut_item'] = prior['prompt']
    prior['prompt_condition'] = 'prior_work'
    prior['n_words'] = prior['response'].apply(lambda x: count_words(x))
    prior_df = prior[['aut_item', 'prompt_condition', 'n_words']]
    mydf = pd.concat([prior_df, wdf])
    summary_stats = mydf.groupby('prompt_condition')['n_words'].agg(['count', 'mean', 'std'])
    logging.info(str(summary_stats))
    make_n_words_graph(mydf)


if __name__ == '__main__':
    main()