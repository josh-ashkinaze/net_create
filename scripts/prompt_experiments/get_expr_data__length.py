import numpy as np
from experiment_class import PromptExperiment
import pandas as pd
import json
import argparse


def run(trials):
    params = {}

    # My items
    my_items = pd.read_csv("../../data/chosen_aut_items.csv")["aut_item"].tolist()

    # Mean WC (used for one prompt)
    example_df = pd.read_csv("../../data/prior_responses.csv")
    my_item_df = example_df.query(f"prompt in {my_items}")
    mean_word_count = round(np.mean(my_item_df['response'].apply(lambda x: len(x.split()))))

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
        'prompts':{key: all_prompts[key] for key in {'zero_shot', 'zero_shot_limit_length'}},
        'example_df':example_df,
        "title": "length",
        "n_uses": 5,
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
    parser.add_argument("--trials", type=int, default=15,
                        help="Number of trials")
    args = parser.parse_args()
    run(args.trials)
