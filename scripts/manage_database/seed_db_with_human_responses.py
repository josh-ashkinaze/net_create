"""
Author: Joshua Ashkinaze
Date: 2023-05-07

Description: This script seeds the trials table with human seed responses. It includes logic to
wait for the table to be ready to accept rows.
"""

import csv
import logging
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import os

import tenacity
from tenacity import retry, wait_fixed

# Add a large number of worlds to the table so that we can run the experiment for a long time
WORLDS = 200

logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

class TableNotFoundError(Exception):
    pass

# If you delete and remake the table in BigQuery, you'll need to wait a bit to add rows to the table, hence
# the sleep and retry logic below.
@retry(wait=wait_fixed(300), stop=tenacity.stop_after_attempt(12), retry_error_callback=lambda x: logging.info(x))
def seed_database():
    logging.info(f"Attempting to seed database with worlds: {WORLDS}")

    key_path = "../../secrets/google_creds.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    dataset = client.dataset("net_expr")
    logging.info("Dataset reference created")

    table = dataset.table("trials")
    logging.info("Table reference created")

    # Read the CSV file and insert the rows into the BigQuery table
    with open("../../data/seed_human_responses.csv", newline="") as csvfile:
        logging.info("Trying to seed human responses")
        reader = csv.DictReader(csvfile)
        response_date = datetime(2022, 1, 1)

        for row in reader:
            # Create a dictionary with the required BigQuery fields and their respective values
            for world in range(WORLDS):
                bq_row = {
                    "item": row["aut_item"],
                    "response_id": f"{row['response_id']}_world{world}",
                    "participant_id": "seed",
                    "condition_order": 1,
                    "response_text": row["response"],
                    "response_date": response_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "condition": row["condition"],
                    "world": world,
                    "init_array": [],
                    "ranked_array": []
                }
                logging.info(f"Attempting to insert row: {bq_row}")

                # Insert the row into the BigQuery table
                try:
                    errors = client.insert_rows_json(table, [bq_row])
                    if not errors:
                        logging.info(f"Row has been added: {bq_row}")
                    else:
                        logging.info(f"Encountered errors while inserting rows: {errors}")
                except Exception as e:
                    error_msg = str(e)
                    logging.info(f"Encountered exception: {error_msg}. Will sleep for 5 minutes and retry.")
                    if "Table" in error_msg and "not found" in error_msg:
                        raise TableNotFoundError(error_msg)
                    else:
                        logging.info("Will retry because of exception {}".format(error_msg))
                        raise e

            # Increment the response_date by 1 minute
            response_date += timedelta(minutes=1)
    logging.info("Finished seeding human responses")

if __name__ == "__main__":
    seed_database()
