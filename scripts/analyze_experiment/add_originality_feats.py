import pandas as pd
import requests
import tenacity
import os
import pickle
import argparse
import logging

class RateLimitError(Exception):
    pass

@tenacity.retry(stop=tenacity.stop_after_attempt(10),
                wait=tenacity.wait_fixed(60),
                retry=tenacity.retry_if_exception_type(RateLimitError),
                reraise=True)
def score_aut_responses(response_tupples):
    base_url = "https://openscoring.du.edu/llm"
    model = "gpt-davinci-paper_alpha"
    input_type = "csv"
    elab_method = "none"

    input_params = []
    for prompt, answer, response_id in response_tupples:
        input_params.append(f"{prompt},{answer}")

    input_str = "&".join([f"input={x}" for x in input_params])

    url = f"{base_url}?model={model}&{input_str}&input_type={input_type}&elab_method={elab_method}"

    response = requests.get(url, headers={'accept': 'application/json'})

    if response.status_code == 200:
        data = response.json()
        scores = data["scores"]

        result = pd.DataFrame(scores, columns=["prompt", "response", "originality"])
        result["response_id"] = [tup[2] for tup in response_tupples]
        return result
    elif response.status_code == 429:
        raise RateLimitError("Rate limit error")
    else:
        return None

def batch_response_tupples(response_tupples, n):
    for i in range(0, len(response_tupples), n):
        yield response_tupples[i:i + n]

def main(debug=True):
    print("Starting")
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
    logging.info("Debug mode: {}".format(debug))
    if debug:
        scores_fn = "../../data/experiment_data/experiment_aut_scores_debug.csv"
    else:
        scores_fn = "../../data/experiment_data/experiment_aut_scores.csv"

    if (not os.path.exists(scores_fn)) or (debug):
        expr_data = pd.read_csv("../../data/experiment_data/data_clean_with_elab.csv")
        if debug: expr_data = expr_data.head(10)
        item_responses_tupples = [(row['item'], row['response_text'], row['response_id']) for _, row in expr_data.iterrows()]
        logging.info(expr_data.head(5))
        logging.info(f"Number of items to score: {len(item_responses_tupples)}")

        scores = []
        counter = 0
        for batch in batch_response_tupples(item_responses_tupples, 5):
            logging.info("Scoring batch {} of {}".format(counter, len(item_responses_tupples) // 5))
            score = score_aut_responses(batch)
            scores.append(score)
            counter += 1
        scores = pd.concat(scores, ignore_index=True)
        scores.to_csv(scores_fn, index=False)

        logging.info(f"Number of items scored: {len(scores)}")

        response2score = {row['response_id']: row['originality'] for _, row in scores.iterrows()}
        with open("../../data/experiment_data/response2score.pkl", 'wb') as f:
            pickle.dump(response2score, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", action="store_true")
    args = parser.parse_args()
    main(debug=args.d)
