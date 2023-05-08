import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


class RateLimitError(Exception):
    pass


def score_aut_responses(response_tupples):
    """Score a list of response tupples (prompt, response) using the OpenScoring API."""
    base_url = "https://openscoring.du.edu/llm"
    model = "gpt-davinci-paper_alpha"
    input_type = "csv"
    elab_method = "none"

    input_params = []
    for prompt, answer in response_tupples:
        input_params.append(f"{prompt},{answer}")

    input_str = "&".join([f"input={x}" for x in input_params])

    url = f"{base_url}?model={model}&{input_str}&input_type={input_type}&elab_method={elab_method}"

    response = requests.get(url, headers={'accept': 'application/json'})

    if response.status_code == 200:
        data = response.json()
        scores = data["scores"]

        result = pd.DataFrame(scores, columns=["prompt", "response", "originality"])
        return result
    elif response.status_code == 429:  # Assuming 429 is the rate limit error code
        raise RateLimitError("Rate limit error")
    else:
        logging.info(f"Error: {response.status_code}")
        return None


def process_aut_batch(batch):
    scores = []
    for x in batch:
        try:
            data = {'aut_item': x[0], 'response': x[1]}
            score = score_aut_responses([data])
            scores.append(score)
        except Exception as e:
            logging.error(f"Failed to score response {x[1]} for item {x[0]}: {e}")
    return scores


def batch_score_responses(response_tupples, batch_size=30, max_workers=4):
    logging.info("Logging scores for responses with batch size of {}".format(batch_size))
    scores = []
    total_batches = math.ceil(len(response_tupples) / batch_size)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch_futures = []

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(response_tupples))
            current_batch = response_tupples[start_idx:end_idx]
            batch_futures.append(executor.submit(process_aut_batch, current_batch))

        for future in batch_futures:
            scores.extend(future.result())

    logging.info("Finished logging scores for responses.")
    return pd.DataFrame(scores)


def make_aesthetic():
    """Make Seaborn look clean, like ggplot"""
    sns.set(style='white', context='poster', font_scale=0.9)
    try:
        plt.rcParams.update({'font.family': 'Arial'})
    except:
        pass
    sns.set_palette(sns.color_palette('dark'))
    sns.despine()
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.titlesize'] = 22