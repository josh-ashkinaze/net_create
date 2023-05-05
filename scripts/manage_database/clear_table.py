import logging
from google.cloud import bigquery
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    key_path = "../../creds/netcreate-0335ce05e7ff.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    dataset_id = "net_expr"
    table_id = "trials"

    # Perform the delete operation
    query = f"DELETE FROM `{credentials.project_id}.{dataset_id}.{table_id}` WHERE TRUE"
    job = client.query(query)
    job.result()

    logger.info(f"All rows deleted from {dataset_id}.{table_id}")

if __name__ == "__main__":
    main()
