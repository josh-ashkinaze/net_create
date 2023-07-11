from google.cloud import bigquery
from google.oauth2 import service_account


def main():
    key_path = f"../../secrets/google_creds.json"
    credentials = service_account.Credentials.from_service_account_file(key_path,
                                                                        scopes=[
                                                                            "https://www.googleapis.com/auth/cloud-platform"], )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    query = """SELECT *, p.dt as person_consent_date, t.response_date as trial_date, f.dt as graded_date
FROM `net_expr.trials` AS t
FULL OUTER JOIN `net_expr.responses` AS f
  ON t.response_id = f.response_id
FULL OUTER JOIN `net_expr.person` AS p
  ON t.participant_id = p.participant_id
  WHERE 
    p.is_test is FALSE AND p.participant_id != 'seed'
"""
    query_job = client.query(query)
    full_join_df = query_job.to_dataframe()
    columns_to_drop = ['response_id_1', 'participant_id_1', 'dt', 'dt_1', 'graded_date', 'is_test']
    df_cleaned = full_join_df.drop(columns=columns_to_drop)
    df_cleaned.to_csv("../../data/expr_data.csv")

if __name__ == "__main__":
    main()
