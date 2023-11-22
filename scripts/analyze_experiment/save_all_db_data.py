"""
Author: Joshua Ashkinaze
Date: 2023-11-17

Description: Gets ALL data and stores as CSV so we can shut down DB
"""

from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np 
import os
import ast
from datetime import datetime

def query_database(client, query):
    query_job = client.query(query)
    df = query_job.to_dataframe()
    return df 

def fetch_experiment_data(client, table):
    print(f"Fetching data for table: {table}")
    
    first_query = f"SELECT * FROM `net_expr.{table}`"
    df = query_database(client, first_query)
    
    # Prefixing filename with the current datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time}_final_fb_view_{table}.csv"
    df.to_csv(filename)
    print(f"Data stored in file: {filename}")
    return df

def main():
    key_path = "../../secrets/google_creds.json"  # Adjust the path as needed
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    for table in ['trials', 'responses', 'person', 'feedback']:
        fetch_experiment_data(client, table)

if __name__ == "__main__":
    main()
