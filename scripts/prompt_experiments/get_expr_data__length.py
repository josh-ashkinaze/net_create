from experiment_class import PromptExperiment
import pandas as pd
import random
import json

def run():
    example_df = pd.read_csv("../../data/prior_responses.csv")
    random_seed = 416
    random.seed(random_seed)

    api_key_path = "../../creds/openai_creds.json"
    with open(api_key_path, "r") as f:
        creds = json.load(f)
    API_KEY = creds["api_key"]

    prompts = {
        "zero_shot": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea, which is an idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. List [N] creative uses for [OBJECT_NAME].",
        "zero_shot_limit_length": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. List [N] creative uses for [OBJECT_NAME]. Make sure each response is 12 words."
    }
    aut_items = pd.read_csv("../../data/chosen_aut_items.csv")["aut_item"].tolist()
    n_uses = 4

    grid_search = {
        "temperature": [0.6, 0.7, 0.8],
        "frequency_penalty": [1, 1.5],
        "presence_penalty": [1, 1.5],
        "n_examples": [4]
    }
    n_trials_per_combo = 1
    prompt_experiment = PromptExperiment(api_key=API_KEY,
                                         title="length",
                                         n_uses =4,
                                         prompts=prompts,
                                         aut_items=aut_items,
                                         example_df=example_df,
                                         random_seed=random_seed)
    prompt_experiment.run(n_trials_per_combo=n_trials_per_combo, grid_search=grid_search)

if __name__ == "__main__":
    run()