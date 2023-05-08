"""
Author: Joshua Ashkinaze
Date: 2023-05-07

Description: This script selects a sample of prior responses (25th to 75th percentile for creativity) to be added to the database,
 and then assigns these responses to conditions
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
        "f_l": 4,
        "f_u": 4,
        "m_l": 2,
        "m_u": 2
    }

    # Read in the data
    df = pd.read_csv("../../data/prior_responses.csv")

    # Get Q2-Q3 responses
    df['creativity_quartile'] = pd.qcut(df['target'], 4, labels=False)
    df['creativity_quartile'] = df['creativity_quartile'] + 1
    df = df[df['creativity_quartile'].isin([2,3])]

    # Filter for my items
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
    assignments_df = assignments_df.rename(columns={'prompt': 'aut_item'})
    assignments_df = assignments_df[['aut_item', 'condition', 'response', 'target']]
    assignments_df['response_id'] = [f"human_seed{i}" for i in range(len(assignments_df))]

    # Save the result
    assignments_df.to_csv("../../data/seed_human_responses.csv", index=False)
    logging.info("All good, got the seed human responses.")

    logging.info(f"Created human_responses.csv with {len(assignments_df)} rows.")

if __name__ == '__main__':
    main()
