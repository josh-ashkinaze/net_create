import logging
from google.cloud import bigquery
from google.oauth2 import service_account
import os

# Set up logging
LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO,
                    format=LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

def delete_rows(credentials, dataset_id, table_id):
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    # Get the schema of the existing table
    table_ref = bigquery.DatasetReference(credentials.project_id, dataset_id).table(table_id)
    table = client.get_table(table_ref)
    schema = table.schema

    # Delete duplicate rows where response_text and world are the same, and keep the latest one
    delete_query = f"""
    DELETE FROM `{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`
    WHERE (response_text, world, response_date) NOT IN (
        SELECT AS STRUCT response_text, world, MAX(response_date) AS max_response_date
        FROM `{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}`
        WHERE DATETIME_DIFF(CURRENT_DATETIME(), response_date, MINUTE) > 30
        GROUP BY response_text, world
    )
    """
    query_job = client.query(delete_query)
    rows_affected = query_job.result().total_rows
    logging.info(f"{rows_affected} rows with duplicate response_text and world deleted from {dataset_id}.{table_id}")

if __name__ == "__main__":
    key_path = "../../secrets/google_creds.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    dataset_id = "net_expr"
    table_id = "trials"
    delete_rows(credentials, dataset_id, table_id)
