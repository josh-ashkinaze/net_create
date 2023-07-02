"""
Author: Joshua Ashkinaze
Date: 2023-07-01

Description
There's a query that happens a lot -- getting the correct items for the correct world. So I tested if different
partitioning and clustering strategies would improve the performance of this query. Results:

------------------------------------------------------------------------------------------------------------------------
         default   clustered        diff
count  100.000000  100.000000  100.000000
mean     0.995990    0.586469    0.409521
std      0.167725    0.139230    0.200954
min      0.817162    0.435082   -0.176610
25%      0.913093    0.493339    0.335339
50%      0.958760    0.543424    0.408134
75%      1.044178    0.621432    0.491370
max      2.265925    1.167769    1.477909

------------------------------------------------------------------------------------------------------------------------
 WITH current_world AS (
    SELECT FLOOR(COUNT(*) / 20) as world_value
    FROM `{client.project}.{dataset_id}.{table_id}`
    WHERE is_test IS FALSE
    AND participant_id != 'seed'
    AND item = '{item}'
    AND condition = '{condition}'
)

SELECT response_text, response_id, response_date
FROM (
    SELECT response_text, response_id, MAX(response_date) as response_date
    FROM `{client.project}.{dataset_id}.{table_id}`, current_world
    WHERE item = '{item}' AND condition = '{condition}' AND is_test = False AND world = current_world.world_value
    GROUP BY response_text, response_id
) AS subquery
ORDER BY response_date DESC
LIMIT {item_limit}

"""

import logging
import os
import time
import pandas as pd
import random
from google.cloud import bigquery
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, filemode='w',
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def create_partitioned_clustered_table(client, dataset_id, table_id, new_table_id):
    # Create a new partitioned and clustered table using a query
    sql = f"""
    CREATE TABLE `{client.project}.{dataset_id}.{new_table_id}`
    PARTITION BY DATE(response_date)
    CLUSTER BY item, condition
    AS SELECT * FROM `{client.project}.{dataset_id}.{table_id}`
    """
    query_job = client.query(sql)
    query_job.result()  # Wait for the query to complete

    logging.info(
        f"Created a new partitioned and clustered table {dataset_id}.{new_table_id} and copied data from {dataset_id}.{table_id}")


def get_random_query_values(client, dataset_id, table_id):
    item_query = f"SELECT DISTINCT item FROM `{client.project}.{dataset_id}.{table_id}`"
    condition_query = f"SELECT DISTINCT condition FROM `{client.project}.{dataset_id}.{table_id}`"

    item_query_job = client.query(item_query)
    condition_query_job = client.query(condition_query)

    items = [row.item for row in item_query_job.result()]
    conditions = [row.condition for row in condition_query_job.result()]

    random_item = random.choice(items)
    random_condition = random.choice(conditions)

    return random_item, random_condition


def run_query_and_log_time(client, dataset_id, table_id, item, condition, item_limit):
    query = f"""
            WITH current_world AS (
                SELECT FLOOR(COUNT(*) / 20) as world_value
                FROM `{client.project}.{dataset_id}.{table_id}`
                WHERE is_test IS FALSE
                AND participant_id != 'seed'
                AND item = '{item}'
                AND condition = '{condition}'
            )

            SELECT response_text, response_id, response_date
            FROM (
                SELECT response_text, response_id, MAX(response_date) as response_date
                FROM `{client.project}.{dataset_id}.{table_id}`, current_world
                WHERE item = '{item}' AND condition = '{condition}' AND is_test = False AND world = current_world.world_value
                GROUP BY response_text, response_id
            ) AS subquery
            ORDER BY response_date DESC
            LIMIT {item_limit}
    """
    start_time = time.time()
    query_job = client.query(query)
    query_job.result()
    elapsed_time = time.time() - start_time
    logging.info(f"Query execution time for table {dataset_id}.{table_id}: {elapsed_time:.2f} seconds")
    return elapsed_time

def main():
    key_path = "../../secrets/google_creds.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    dataset_id = "net_expr"
    table_id = "trials"
    new_table_id = "trials_partitioned_clustered2"

    try:
        create_partitioned_clustered_table(client, dataset_id, table_id, new_table_id)
    except Exception as e:
        logging.error(e)
    table_times = []
    for i in range(100):
        item, condition = get_random_query_values(client, dataset_id, table_id)
        item_limit = 6

        default_table_time = run_query_and_log_time(client, dataset_id, table_id, item, condition, item_limit)
        clustered_table_time = run_query_and_log_time(client, dataset_id, new_table_id, item, condition, item_limit)
        table_times.append({'default': default_table_time, 'clustered': clustered_table_time})
        logging.info(f"Performance improvement: {default_table_time / clustered_table_time:.2f} times")
    df = pd.DataFrame(table_times)
    df['diff'] = df['default'] - df['clustered']
    logging.info(df.describe())


if __name__ == "__main__":
    main()