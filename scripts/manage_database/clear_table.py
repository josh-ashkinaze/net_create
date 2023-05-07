import logging
from google.cloud import bigquery
from google.oauth2 import service_account
import os

# Set up logging
LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(filename=f'{os.path.basename(__file__)}.log', level=logging.INFO,
                    format=LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

def truncate_table(credentials, dataset_id, table_id):
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    # Delete all rows from the table
    table_ref = client.dataset(dataset_id).table(table_id)
    delete_query = f"DELETE FROM `{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}` WHERE 1=1"
    query_job = client.query(delete_query)
    query_job.result()

    logging.info(f"All rows truncated from {dataset_id}.{table_id}")

if __name__ == "__main__":
    key_path = "../../creds/netcreate-0335ce05e7ff.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    dataset_id = "net_expr"
    table_id = "trials"
    truncate_table(credentials, dataset_id, table_id)
