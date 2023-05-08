""""
Date: 03/12/2023

Description: This is a Flask web application that runs a web experiment.

* Participants go through a series of trials, where they are asked to respond to various prompts under different conditions.
* The application collects participants' responses and stores them in a Google BigQuery database.
* The experiment counterbalances across different conditions and stimuli
* The experiment has four routes:  the consent form, the start of the experiment, the trials themselves, and the thank you page.

# Condition defintions:
# - 6 human-only ideas
# - f, l: few (2) AI ideas / many (4) human ideas, labeled source
# - f, u: few (2) AI ideas / many (4) human ideas, unlabled source
# - m, l: many (4) AI ideas / few (2) human ideas, labled source
# - m, u: many (4) AI ideas / few (2) human ideas, unlabled source

"""

from google.cloud import bigquery
from flask import Flask, render_template, request, redirect, url_for, session
from google.oauth2 import service_account
import uuid
import pandas as pd
import time
from datetime import datetime
import random
import os
import sys
from render_graph import make_graphs  # import the comparison_graph function from render_graph.py


# Figure out if we're running locally or on Heroku. This will matter for file paths.
if 'DYNO' in os.environ:
    is_local = False
    file_prefix = ""
else:
    is_local = True
    file_prefix = "../"

# EXPERIMENT PARAMETERS
####################
SOURCE_LABEL = "For this object, we also asked AI to come up with ideas! "
CONDITIONS = {
    'h': {'n_human': 6, 'n_ai': 0, 'label':False},
    'f_l': {'n_human': 4, 'n_ai': 2, 'label':True},
    'f_u': {'n_human': 4, 'n_ai': 2, 'label': False},
    'm_l': {'n_human': 2, 'n_ai': 4, 'label': True},
    'm_u': {'n_human': 2, 'n_ai': 4, 'label': False},
}
ITEMS = pd.read_csv(file_prefix + "data/chosen_aut_items.csv")['aut_item'].unique().tolist()
AI_IDEAS_DF = pd.read_csv(file_prefix + "data/ai_responses.csv")
####################




# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'k'

# BigQuery credentials
key_path = file_prefix + "creds/netcreate-0335ce05e7ff.json"
credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

# BigQuery connection
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
dataset = client.dataset("net_expr")
table = dataset.table("trials")


@app.route("/")
def consent_form():
    """Consent form"""
    return render_template('consent_form.html')


@app.route("/start-experiment")
def start_experiment():
    """
    Start the experiment.

    Let's first create a participant ID, counterbalance conditions, and counterbalance items.
    Store these items in the global variable called `TEMP'.

    Then, we start the experiment by calling `render_trial(conditon_no=0)'.
    From there, `render_trial' will recursively call itself to display all trials until the experiment is done.

    """
    session['participant_id'] = str(uuid.uuid4())
    item_order = list(ITEMS)
    random.shuffle(item_order)
    condition_order = list(CONDITIONS.keys())
    random.shuffle(condition_order)
    session['condition_order'] = condition_order
    session['item_order'] = item_order
    session['responses'] = []
    return redirect(url_for('render_trial', condition_no=0, method="GET"))


@app.route("/render_trial/<int:condition_no>", methods=['GET', 'POST'])
def render_trial(condition_no):
    """
    Recursively handles the render_trial route for a particular condition_no.

    The idea is that in a temp dictionary, we store the participant ID, the condition order, and the item order.
    Then, we keep calling this function with the next condition_no -- which indexes items and conditions --
    until we've gone through all conditions.

    The logic is as follows:

    IF the current condition_no number is more than the number of items:
        Return the thank_you page since our experiment is done.
    
    ELSE if there are still trials to go:
        1. If the HTTP method is GET (i.e: response not submitted), retrieve the necessary context from the global TEMP
        dict and generate an render_trial instance. Upon submitting a response, this submits a post request.

        2. If the HTTP method is POST (i.e: response was submitted), the function retrieves the participant's response
        text and inserts it into a BigQuery table. Then we make GET request to `render_trial(condition_no+1)' to
        go to the next condition/item.


    Parameters:
    - condition_no (int): the current condition_no

    Returns:
    - Either another instance of render_trial or the thank_you page

    """
    # If the participant has completed all condition_nos, redirect to thank you page
    if condition_no > len(ITEMS)-1:
        return redirect(url_for('thank_you'))
    else:
        pass

    # Retrieve the necessary information from the global TEMP variable
    participant_id = session['participant_id']
    condition = session['condition_order'][condition_no]
    to_label = CONDITIONS[condition]['label']
    human_ideas = CONDITIONS[condition]['n_human']
    ai_ideas = CONDITIONS[condition]['n_ai']
    item = session['item_order'][condition_no]

    # If the HTTP method is GET, render the render_trial template
    if request.method == "GET":
        time.sleep(0.1)
        human_rows = [row['response_text'] for row in list(client.query(
            f"SELECT response_text FROM `net_expr.trials` WHERE (item = '{item}' AND condition = '{condition}') ORDER BY response_date DESC LIMIT {human_ideas}").result())]
        ai_rows = AI_IDEAS_DF.query("aut_item=='{}'".format(item)).sample(ai_ideas)['response'].tolist()
        if to_label:
            human_rows = [row + ' <span style="color: #1F4287;">(Source: <strong>Human</strong>)</span>' for row in
                          human_rows]
            ai_rows = [row + ' <span style="color: #1F4287;">(Source: <strong>A.I</strong>)</span>' for row in ai_rows]

        rows = ai_rows + human_rows
        random.shuffle(rows)
        print("responses", rows)
        return render_template('render_trial.html', item=item, rows=rows, condition_no=condition_no, label=SOURCE_LABEL if to_label else "")

    # If the HTTP method is POST, insert the participant's response into the BigQuery table
    # then increment the condition_no and redirect to the next render_trial
    elif request.method == 'POST':

        # Retrieve the participant's response
        response_text = request.form.get('participant_response')
        session['responses'].append(response_text)
        session.modified = True  # Explicitly mark the session as modified
        print("printing responses", session['responses'])

        # Insert the participant's response into the BigQuery table
        row = {
            "item": item,
            "response_id": str(uuid.uuid4()),
            "participant_id": participant_id,
            "condition_order": condition_no,
            "response_text": response_text,
            "response_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "condition": condition,
            "world":1
        }
        errors = client.insert_rows_json(table, [row])
        if not errors:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))

        # Redirect to next condition_no
        return redirect(url_for('render_trial', condition_no=condition_no + 1, method="GET"))


@app.route("/thank-you")
def thank_you():
    """Thank you page"""
    participant_responses = session['responses']  # example participant_responses
    comparison = "human"  # example comparison
    participant_responses = list(zip(session['item_order'], session['responses']))
    participant_conditions = session['condition_order']
    human_graph, ai_graph, human_ai_graph = make_graphs(participant_responses, participant_conditions, file_prefix)

    # Generate the human comparison graph

    # Generate the AI comparison graph
    return render_template('thank_you.html', img_base64_human=human_graph, img_base64_ai=ai_graph, img_base64_human_ai=human_ai_graph)


if __name__ == '__main__':
    if is_local:
        app.run(port=5027, debug=True)
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(host="0.0.0.0", port=port)
