"""
Author: Joshua Ashkinaze
Date: 2023-05-07

Description: This script selects a sample of prior responses to be added to the database, assigns these responses to conditions
"""

import pandas as pd
import random
import logging
import os

def main():
    logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
    random.seed(416)

    n_items_per_condition = {
        "h": 6,
        "f_l": 2,
        "f_u": 2,
        "m_l": 4,
        "m_u": 4
    }

    # Read in the data and filter to only include the relevant prompts
    df = pd.read_csv("../../data/prior_responses.csv")
    my_items = pd.read_csv("../../data/chosen_aut_items.csv")['aut_item'].tolist()
    df = df[df['prompt'].isin(my_items)]

    # Initialize a list to hold the assignments
    assignments = []

    for prompt in my_items:
        # Filter data for current prompt
        prompt_df = df[df['prompt'] == prompt]

        for condition, n_items in n_items_per_condition.items():
            # Sample n_items from prompt_df for each condition
            sampled_df = prompt_df.sample(n_items, random_state=417)

            # Remove sampled rows from prompt_df
            prompt_df = prompt_df.drop(sampled_df.index)

            # Add condition column to sampled_df
            sampled_df['condition'] = condition

            # Append sampled_df to assignments
            assignments.append(sampled_df)

    # Concatenate all the DataFrames in assignments
    assignments_df = pd.concat(assignments)

    # Save the result
    assignments_df.to_csv("../../data/seed_human_responses.csv", index=False)
    logging.info("All good, got the seed human responses.")

    logging.info(f"Created human_responses.csv with {len(assignments_df)} rows.")

if __name__ == '__main__':
    main()
