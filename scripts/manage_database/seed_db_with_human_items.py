import csv
import logging
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_database():
    key_path = "../../creds/netcreate-0335ce05e7ff.json"
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    dataset = client.dataset("net_expr")
    table = dataset.table("trials")

    # Read the CSV file and insert the rows into the BigQuery table
    with open("../../data/seed_human_responses.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        response_date = datetime(2022, 1, 1)

        for row in reader:
            # Create a dictionary with the required BigQuery fields and their respective values
            bq_row = {
                "item": row["item"],
                "response_id": "test",
                "participant_id": "test",
                "condition_order": 1,
                "response_text": row["response"],
                "response_date": response_date.strftime("%Y-%m-%d %H:%M:%S"),
                "condition": row["condition"],
            }

            # Insert the row into the BigQuery table
            errors = client.insert_rows_json(table, [bq_row])
            if not errors:
                logger.info(f"Row has been added: {bq_row}")
            else:
                logger.error(f"Encountered errors while inserting rows: {errors}")

            # Increment the response_date by 1 minute
            response_date += timedelta(minutes=1)

if __name__ == "__main__":
    seed_database()
