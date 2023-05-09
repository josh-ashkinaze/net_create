"""
Author: Joshua Ashkinaze
Date: 2023-05-07

Description: This script clears all rows from trials table
"""

import logging
from google.cloud import bigquery
from google.oauth2 import service_account
import os

# Set up logging
logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO, filemode='w',
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def main():
    key_path = "../../secrets/google_creds.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    dataset_id = "net_expr"
    table_id = "trials"

    # Get the schema of the existing table
    table_ref = bigquery.DatasetReference(credentials.project_id, dataset_id).table(table_id)
    table = client.get_table(table_ref)
    schema = table.schema

    # Delete the table
    client.delete_table(table_ref)

    # Create a new empty table with the same schema
    new_table = bigquery.Table(table_ref, schema=schema)
    client.create_table(new_table)

    logging.info(f"All rows cleared from {dataset_id}.{table_id}")

if __name__ == "__main__":
    main()
