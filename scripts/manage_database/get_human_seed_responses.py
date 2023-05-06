"""
Description: This script seeds the database with human responses to the AUT items

Author: Joshua Ashkinaze

Date: 05-05-2023
"""
import pandas as pd
import random
import logging
import os

def main():
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    random.seed(416)

    my_items = ["box", "fork", "lightbulb", "spoon", "table"]

    # Number of human responses needed:
    # H = 6
    # F, L: 2
    # F, U: 2
    # M, L: 4
    # M, U: 4
    n_items_per_condition = [6, 2, 2, 4, 4]

    # Read in the data and filter to only include the relevant prompts
    df = pd.read_csv("../../data/prior_responses.csv")
    df = df[df['prompt'].isin(my_items)]
    df = df.sample(frac=1, random_state=416)

    # Initialize a list to hold the assignments
    assignments = []

    # Loop over each prompt
    for prompt in my_items:
        # Get a list of all responses for this prompt
        responses = df[df['prompt'] == prompt]['response'].unique().tolist()
        scores = df[df['prompt'] == prompt]['target'].unique().tolist()

        for i, n_items in enumerate(n_items_per_condition):
            condition = ""
            if i == 0:
                condition = "h"
            elif i == 1:
                condition = "f_l"
            elif i == 2:
                condition = "f_u"
            elif i == 3:
                condition = "m_l"
            else:
                condition = "m_u"
            for j in range(n_items):
                response = responses.pop(0)
                score = scores.pop(0)
                assignments.append({'item': prompt, 'response': response.lower(), 'score':score, 'condition': condition})

    # Convert the list of dictionaries to a DataFrame
    assignments_df = pd.DataFrame(assignments)
    assignments_df.to_csv("../../data/seed_human_responses.csv")
    logging.info("All good, got the seed human responses.")


    logging.info(f"Created human_responses.csv with {len(assignments_df)} rows.")

if __name__ == '__main__':
    main()
