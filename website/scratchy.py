from google.cloud import bigquery
from google.oauth2 import service_account

key_path = "../creds/netcreate-0335ce05e7ff.json"
credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = bigquery.Client(credentials=credentials, project=credentials.project_id)
dataset = client.dataset("net_expr")
table = dataset.table("trials")

row = {
    "id": "unique_id",
    "participant_id": "participant1",
    "response_text": "This is a response",
    "response_date": "2022-02-21",
    "condition": 1,
    "condition_order": 1,
    "source": "human"
}
errors = client.insert_rows_json(table, [row])
if errors == []:
    print("New rows have been added.")
else:
    print("Encountered errors while inserting rows: {}".format(errors))
