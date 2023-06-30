import logging
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from google.api_core.exceptions import BadRequest
import argparse

LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.FileHandler(f'{os.path.basename(__file__)}.log')
handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S'))
root_logger.addHandler(handler)

@retry(retry=retry_if_exception_type(BadRequest),
       stop=stop_after_attempt(50),
       wait=wait_exponential(multiplier=5, min=1, max=5400),
       before_sleep=before_sleep_log(logging.getLogger(__name__), logging.ERROR))
def execute_query(client, clear_query):
    query_job = client.query(clear_query)
    query_job.result()

def clear_rows(credentials, dataset_id, table_ids, remove_seeds):
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    for table_id in table_ids:
        table_ref = bigquery.DatasetReference(credentials.project_id, dataset_id).table(table_id)
        table = client.get_table(table_ref)
        schema = table.schema

        if table_id == 'trials' and not remove_seeds:
            clear_query = f"""
            DELETE FROM `{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`
            WHERE NOT REGEXP_CONTAINS(participant_id, r'seed')
            """
        else:
            clear_query = f"""
            DELETE FROM `{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`
            WHERE TRUE
            """
        execute_query(client, clear_query)
        logging.info(f"All rows cleared from {dataset_id}.{table_id}, except those where participant_id contains 'seed' for 'trials' table (if remove_seeds is False)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clear rows from Google BigQuery tables.')
    parser.add_argument('--remove_seeds', action='store_true',
                        help='Whether to remove rows where participant_id contains "seed" in the "trials" table.')
    args = parser.parse_args()

    confirm = input("Type 'confirm' to proceed with deleting rows: ")
    if confirm.lower() != 'confirm':
        print("Operation cancelled.")
        exit()

    key_path = "../../secrets/google_creds.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    dataset_id = "net_expr"
    table_ids = ["trials", "person", "responses", 'feedback']
    clear_rows(credentials, dataset_id, table_ids, args.remove_seeds)
    logging.info(f"All rows cleared appropiately from {dataset_id}.{table_ids}")
