import numpy as np
from experiment_class import PromptExperiment
import pandas as pd
import json
import argparse


def run(trials):
    params = {}

    # Mean WC (used for one prompt)
    example_df = pd.read_csv("../../data/prior_responses.csv")
    mean_word_count = int(np.mean(example_df['response'].apply(lambda x: len(x.split()))))

    # Get OpenAI keys
    with open("../../secrets/openai_creds.json", "r") as f:
        api_key = json.load(f)["api_key"]

    # Get prompts
    with open('prompt_list.json') as f:
        all_prompts = json.load(f)['prompts']

    # Fill in experiment details

    # Define parameters
    params = {
        'api_key':api_key,
        'aut_items': pd.read_csv("../../data/chosen_aut_items.csv")["aut_item"].tolist(),
        'prompts':{key: all_prompts[key] for key in {'zero_shot', 'zero_shot_limit_length', 'implicit_transfer', 'explicit_transfer'}},
        'example_df':example_df,
        "title": "stransfer",
        "n_uses": 5,
        "n_examples":5,
        "by_quartile": True,
        "n_trials": args.trials,
        "llm_params": {
            "temperature": [.65, .70, .75, 0.80],
            "frequency_penalty": [1],
            "presence_penalty": [1],
        }
    }
    params['prompts']['zero_shot_limit_length'] = params['prompts']['zero_shot_limit_length'].replace(
        "[MEAN_HUMAN_WORDS]", str(mean_word_count))

    prompt_experiment = PromptExperiment(**params)
    prompt_experiment.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trials per combination of prompt, item, and example count.")
    args = parser.parse_args()
    run(args.trials)
