import pandas as pd
import requests
import tenacity
import os
import pickle
import argparse
import logging
from urllib.parse import quote



class RateLimitError(Exception):
    pass

@tenacity.retry(stop=tenacity.stop_after_attempt(10),
                wait=tenacity.wait_fixed(60),
                retry=tenacity.retry_if_exception_type(RateLimitError),
                reraise=True)
def score_aut_responses(response_tuples):
    base_url = "https://openscoring.du.edu/llm"
    model = "gpt-davinci-paper_alpha"
    input_type = "csv"
    elab_method = "none"

    input_params = []
    for prompt, answer, participant_id in response_tuples:
        # Wrap prompt and answer in double quotes to handle commas
        input_params.append(quote(f'"{prompt}","{answer}",{participant_id}'))

    input_str = "&".join([f"input={x}" for x in input_params])

    url = f"{base_url}?model={model}&{input_str}&input_type={input_type}&elab_method={elab_method}"

    response = requests.post(url, headers={'accept': 'application/json'})

    if response.status_code == 200:
        data = response.json()
        scores = data["scores"]
        result = pd.DataFrame(scores)
        result = result.rename(columns = {'participant_id':'response_id'})
        return result
    elif response.status_code == 429:
        logging.info("rate limit error")
        raise RateLimitError("Rate limit error")
    else:
        return None


def fetch_missed_scores():
    scores_fn = "../../data/experiment_data/experiment_aut_scores.csv"
    attempts = 0
    while True or attempts < 50:
        attempts += 1
        logging.info(f"Attempt {attempts}")
        scores_df = pd.read_csv(scores_fn)
        response_ids_in_scores = set(scores_df['response_id'])
        expr_data = pd.read_csv("../../data/experiment_data/data_clean_with_elab.csv")
        experiment_response_ids = set(expr_data['response_id'])
        missing_response_ids = experiment_response_ids - response_ids_in_scores
        if missing_response_ids:
            logging.info(f"Number of missing items to score: {len(missing_response_ids)}")
            item_responses_tupples = [(row['item'], row['response_text'], row['response_id']) for _, row in expr_data.iterrows() if row['response_id'] in missing_response_ids]
            logging.info(item_responses_tupples[:5])
            for batch in batch_response_tupples(item_responses_tupples, 5):
                score = score_aut_responses(batch)
                if score is not None:
                    scores_df = pd.concat([scores_df, score], ignore_index=True)
            scores_df.to_csv(scores_fn, index=False)
            logging.info(f"Number of total items scored: {len(scores_df)}")
            logging.info("Total number of responses in experiment data: {}".format(len(expr_data)))
            response2score = {row['response_id']: row['originality'] for _, row in scores_df.iterrows()}
            with open("../../data/experiment_data/response2score.pkl", 'wb') as f:
                pickle.dump(response2score, f)
        else:
            logging.info("No missing response IDs detected.")
            break
    logging.info("Ending")

def get_scores(debug=True):
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

def batch_response_tupples(response_tupples, n):
    for i in range(0, len(response_tupples), n):
        yield response_tupples[i:i + n]

if __name__ == "__main__":
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch_missed_scores()
    else:
        get_scores(debug=args.d)
