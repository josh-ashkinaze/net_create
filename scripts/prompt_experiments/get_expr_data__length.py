import numpy as np

from experiment_class import PromptExperiment
import pandas as pd
import random
import json
import argparse


def run(n_trials_per_combo=1):
    random_seed = 416

    # Prior responses
    example_df = pd.read_csv("../../data/prior_responses.csv")
    example_df['word_count'] = example_df['response'].apply(lambda x: len(x.split()))
    mean_word_count = int(np.mean(example_df['word_count']))

    # Get OpenAI keys
    api_key_path = "../../creds/openai_creds.json"
    with open(api_key_path, "r") as f:
        creds = json.load(f)
    API_KEY = creds["api_key"]

    # Get prompts
    with open('prompt_list.json') as f:
        all_prompts = json.load(f)['prompts']
    prompts = {key: all_prompts[key] for key in {'zero_shot', 'zero_shot_limit_length'}}
    prompts['zero_shot_limit_length'] = prompts['zero_shot_limit_length'].replace("[MEAN_HUMAN_WORDS]", str(mean_word_count))

    aut_items = pd.read_csv("../../data/chosen_aut_items.csv")["aut_item"].tolist()
    n_uses = 5
    llm_params = {
        "temperature": [.65, .70, .75, 0.80],
        "frequency_penalty": [1],
        "presence_penalty": [1],
    }

    prompt_experiment = PromptExperiment(api_key=API_KEY,
                                         title="length",
                                         n_uses=n_uses,
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
