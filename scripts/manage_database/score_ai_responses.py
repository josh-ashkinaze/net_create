import logging
import os

import pandas as pd

import helpers.helpers as my_utils


def main():
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, format=LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
    logger = logging.getLogger()
    ai_responses_df = pd.read_csv("../../data/ai_responses.csv")
    response_tupples = list(ai_responses_df[['aut_item', 'response']].itertuples(index=False, name=None))
    scored_responses_df = my_utils.batch_score_responses(response_tupples, batch_size=30, logger=logger)
    scored_responses_df.to_csv("../../data/scored_ai_responses.csv", index=False)
    logging.info(f"Scored responses")


if __name__ == "__main__":
    main()
