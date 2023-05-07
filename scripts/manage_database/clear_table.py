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

    # Get a reference to the table
    table_ref = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table_ref)

    # Create a new table with the same schema as the old table
    new_table_id = f"{table_id}_temp"
    new_table_ref = client.dataset(dataset_id).table(new_table_id)
    new_table = bigquery.Table(new_table_ref, schema=table.schema)
    client.create_table(new_table)

    # Delete the old table
    client.delete_table(table)

    # Rename the new table to the original table name
    new_table.table_id = table_id
    client.update_table(new_table, ["table_id"])

    logging.info(f"All rows truncated from {dataset_id}.{table_id}")

if __name__ == "__main__":
    key_path = "../../creds/netcreate-0335ce05e7ff.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    dataset_id = "net_expr"
    table_id = "trials"
    truncate_table(credentials, dataset_id, table_id)
