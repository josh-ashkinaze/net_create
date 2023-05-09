import numpy as np

from experiment_class import PromptExperiment
import pandas as pd
import random
import json
import argparse


def run(n_trials_per_combo=1):
    example_df = pd.read_csv("../../data/prior_responses.csv")
    example_df['word_count'] = example_df['response'].apply(lambda x: len(x.split()))
    mean_word_count = int(np.mean(example_df['word_count']))
    print("Mean word count", mean_word_count)
    random_seed = 416
    random.seed(random_seed)

    with open("../../secrets/openai_creds.json", "r") as f:
        API_KEY = json.load(f)["api_key"]

    prompts = {
        "zero": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea, which is an idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. List [N] creative uses for [OBJECT_NAME].",
        "zero_limit": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. List [N] creative uses for [OBJECT_NAME]. Make sure each response is around {} words.".format(
            mean_word_count),
        "implicit_transfer": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea, which is an idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Here are example creative uses: [EXAMPLES] Based on the examples, list [N] creative uses for [OBJECT_NAME] that sound like the examples.",
        "explicit_transfer": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea, which is an idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Here are example creative uses: [EXAMPLES] Carefully study the examples and their style, then list [N] creative uses for [OBJECT_NAME] that resemble the given examples. Match the style, length, and complexity of the creative ideas in the examples.",

    }
    aut_items = pd.read_csv("../../data/chosen_aut_items.csv")["aut_item"].tolist()
    n_uses = 4

    llm_params = {
        "temperature": [0.6, 0.7, 0.8],
        "frequency_penalty": [1, 1.5],
        "presence_penalty": [1, 1.5]
    }
    prompt_experiment = PromptExperiment(api_key=API_KEY,
                                         title="stransfer",
                                         n_examples=4,
                                         n_uses=4,
                                         prompts=prompts,
                                         aut_items=aut_items,
                                         example_df=example_df,
                                         random_seed=random_seed)
    prompt_experiment.run(n_trials_per_combo=n_trials_per_combo, llm_params=llm_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trials per combination of prompt, item, and example count.")
    args = parser.parse_args()
    run(args.trials)
