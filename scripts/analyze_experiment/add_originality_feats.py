import pandas as pd
import requests
import tenacity  # Don't forget to import tenacity
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import os
import argparse

class RateLimitError(Exception):
    def __init__(self, message, logger=None):
        super().__init__(message)
        if logger:
            logger.info("Rate limit error")
@tenacity.retry(stop=tenacity.stop_after_attempt(5),  # Change the number of attempts as needed
                wait=tenacity.wait_fixed(60),  # Change the wait time between retries as needed
                retry=tenacity.retry_if_exception_type(RateLimitError),
                reraise=True)
def score_aut_responses(response_tupples, logger=None):
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
        raise RateLimitError("Rate limit error", logger)
    else:
        return None


def process_aut_batch(batch, logger=None):
    scores = []
    for x in batch:
        data = {'aut_item': x[0], 'response': x[1]}
        score = score_aut_responses([(data['aut_item'], data['response'])], logger)
        scores.append(score)
    return scores


def batch_score_responses(response_tupples, batch_size=30, max_workers=15, logger=None):
    if logger:
        logging.info("Logging scores for responses with batch size of {}".format(batch_size))
    scores = []
    total_batches = math.ceil(len(response_tupples) / batch_size)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batch_futures = []

        for batch_idx in range(total_batches):
            if logger:
                logging.info("Processing batch {} out of {}".format(batch_idx + 1, total_batches))
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(response_tupples))
            current_batch = response_tupples[start_idx:end_idx]
            batch_futures.append(executor.submit(process_aut_batch, current_batch, logger))

        for future in batch_futures:
            scores.extend(future.result())

    return pd.concat(scores, ignore_index=True)



def main(debug=False):
    if debug:
        scores_fn = "../../data/experiment_data/experiment_aut_scores_debug.csv"
    else:
        scores_fn = "../../data/experiment_data/experiment_aut_scores.csv"

    # If debugging mode or didn't get data yet...
    if (not os.path.exists(scores_fn)) or (debug):
        LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                            datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
        logging.info("Starting to score AUT responses")
        logging.info("Debug mode: {}".format(debug))
        expr_data = pd.read_csv("../../data/experiment_data/data_clean_with_elab.csv")
        if debug: expr_data = expr_data.head(10)
        item_responses_tupples = [(row['item'], row['response_text']) for _, row in expr_data.iterrows()]
        scores = batch_score_responses(item_responses_tupples, logger=logging)
        scores.to_csv(scores_fn, index=False)
        logging.info("There were {} original responses and {} scored responses".format(len(expr_data), len(scores)))
        response2score = {row['response']: row['originality'] for _, row in scores.iterrows()}
        with open("../../data/experiment_data/response2score.pkl", 'wb') as f:
            pickle.dump(response2score, f)
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", action="store_true")
    args = parser.parse_args()
    main(debug=args.d)