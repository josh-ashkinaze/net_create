""""
Date: 05/07/2023

Description: This is a Flask web application that runs a web experiment.

* Participants go through a series of trials, where they are asked to respond to various prompts under different conditions.
* The application collects participants' responses and stores them in a Google BigQuery database.
* The experiment counterbalances across different conditions and stimuli
* The experiment has four routes:  the consent form, the start of the experiment, the trials themselves, and the thank you page.

# Conditions:
# There is a control human-only condition, and then a 2x2 varying by
# LLM exposure (few LLM ideas vs many LLM ideas) and transparency (say source labels or don't)
"""

import json
import os
import random
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from flask import flash, Flask, render_template, request, redirect, url_for, session, jsonify
from google.cloud import bigquery
from google.oauth2 import service_account
from scipy.stats import spearmanr
from urllib.parse import urlencode

############################################################################################################
# SETUP GLOBAL VARIABLES
############################################################################################################

# Figure out if we're running locally or on Heroku. This can matter for file paths.

# Note: Depends on which root you want to run from for whether file_prefix should differ.
# The updaed way I run locally is: `cd net_create  FLASK_APP=website/web.py flask run`
# I switched to this method because this better matches how Heroku runs the app

if 'DYNO' in os.environ:
    is_local = False
    file_prefix = ""
    from render_feedback import make_graphs, calculate_similarity
else:
    is_local = True
    file_prefix = ""
    from website.render_feedback import make_graphs, calculate_similarity

# Initialize the Flask application
app = Flask(__name__)
if is_local:
    flask_secret_key = json.load(open(f"{file_prefix}secrets/flask_secret_key.json", "r"))['session_key']
else:
    flask_secret_key = os.environ['FLASK_SECRET_KEY']

if is_local:
    is_test = True
else:
    is_test = bool(os.environ['IS_TEST'])

app.secret_key = flask_secret_key

# Connect to BQ
if not is_local:
    json_key = json.loads(os.environ['GOOGLE_CREDS'])
    credentials = service_account.Credentials.from_service_account_info(json_key)
else:
    key_path = f"{file_prefix}secrets/google_creds.json"
    credentials = service_account.Credentials.from_service_account_file(key_path,
                                                                        scopes=[
                                                                            "https://www.googleapis.com/auth/cloud-platform"], )
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
dataset = client.dataset("net_expr")
table = dataset.table("trials")

# EXPERIMENT PARAMETERS
N_PER_WORLD = 20
SOURCE_LABEL = "For this object, we also asked AI to come up with ideas! "
CONDITIONS = {'h': {'n_human': 6, 'n_ai': 0, 'label': False}, 'f_l': {'n_human': 4, 'n_ai': 2, 'label': True},
              'f_u': {'n_human': 4, 'n_ai': 2, 'label': False}, 'm_l': {'n_human': 2, 'n_ai': 4, 'label': True},
              'm_u': {'n_human': 2, 'n_ai': 4, 'label': False}, }
ITEMS = pd.read_csv(file_prefix + "data/chosen_aut_items.csv")['aut_item'].unique().tolist()
AI_IDEAS_DF = pd.read_csv(file_prefix + "data/ai_responses.csv")


############################################################################################################
############################################################################################################


# START THE EXPERIMENT
############################################################################################################
@app.route("/")
def consent_form():
    """Consent form"""
    return render_template('consent_form.html')


@app.route("/start-experiment")
def start_experiment():
    """
    Start the experiment.

    1. Ccreate a UUID for participant ID
    2. Counterbalance conditions and counterbalance items.
    3. Save all this stuff to a Flask session object.
    4. Redirect to the first trial.
    """

    # Assign UUID to participant
    if is_local:
        reset_session()  # Add this line
    session['participant_id'] = str(uuid.uuid4())
    session['world'] = get_world()

    creativity_ai = request.args.get('creativitySliderAIValue')
    creativity_human = request.args.get('creativitySliderHumanValue')
    ai_feeling = request.args.get('aiFeelingValue')
    country = request.args.get('countryValue')
    age = request.args.get('ageValue')

    session['referer'] = request.headers.get('Referer')
    session['creativity_ai'] = int(creativity_ai) if creativity_ai != '' else None
    session['creativity_human'] = int(creativity_human) if creativity_human != '' else None
    session['age'] = int(age) if age != '' else None
    session['ai_feeling'] = ai_feeling
    session['country'] = country
    session.modified = True

    # Add participant to the person table
    person_table = dataset.table("person")
    row = {"participant_id": session['participant_id'], "creativity_ai": session['creativity_ai'],
           "creativity_human": session['creativity_human'], "age": session['age'],
           "dt": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), 'ai_feeling': session['ai_feeling'],
           'country': session['country'], 'is_test':is_test, 'referer':session['referer']}

    print(session, is_local)

    errors = client.insert_rows_json(person_table, [row])
    if errors:
        print("Encountered errors while inserting rows: {}".format(errors))
    else:
        print("New row has been added to person.")

    session['item_order'] = random.sample(ITEMS, len(ITEMS))
    session['condition_order'] = random.sample(list(CONDITIONS.keys()), len(CONDITIONS))
    session['responses'] = []
    session['response_ids'] = []
    return redirect(url_for('render_trial', condition_no=0, method="GET"))


@app.route("/render_trial/<int:condition_no>", methods=['GET', 'POST'])
def render_trial(condition_no):
    """
    Recursively handles the render_trial route for a particular condition_no.

    The idea is that in a session object (like a unique dictionary or each participant),
    we store the participant ID, the condition order, and the item order.
    Then, we keep calling this function with the next condition_no -- which indexes the participant's items and condition sequence --
    until we've gone through all trials.

    The logic is as follows:

    IF the current condition_no number is more than the number of items:
        Return the thank_you page since our experiment is done.

    ELSE if there are still trials to go:
        1. If the HTTP method is GET (i.e: response not submitted), retrieve the necessary context from the session
         and generate a render_trial instance. Upon submitting a response, this submits a post request.

        2. If the HTTP method is POST (i.e: response was submitted), retrieve the participant's response
        text and insert it into a BigQuery table. Then we make GET request to `render_trial(condition_no+1)' to
        go to the next condition/item.


    Parameters:
    - condition_no (int): the current condition_no

    Returns:
    - Either another instance of render_trial (if not done) or the thank_you page (if done)

    """
    # If the participant has completed all condition_nos, redirect to thank you page
    if condition_no > len(ITEMS) - 1:
        return redirect(url_for('results', uuid=session['participant_id']) + "?from_uuid=False")

    else:
        pass

    participant_id = session['participant_id']
    condition = session['condition_order'][condition_no]
    to_label = CONDITIONS[condition]['label']
    n_human_ideas = CONDITIONS[condition]['n_human']
    n_ai_ideas = CONDITIONS[condition]['n_ai']
    item = session['item_order'][condition_no]
    world = session['world']

    # If the HTTP method is GET, render the render_trial template
    if request.method == "GET":
        # Get the human ideas and their IDs
        human_result = list(client.query(f"""
                    SELECT response_text, response_id, response_date 
                    FROM (
                        SELECT response_text, response_id, MAX(response_date) as response_date 
                        FROM `net_expr.trials` 
                        WHERE item = '{item}' AND condition = '{condition}' AND world = {world} 
                        GROUP BY response_text, response_id
                    ) AS subquery
                    ORDER BY response_date DESC
                    LIMIT {n_human_ideas}
                """).result())

        human_rows = [row['response_text'] for row in human_result]
        human_ids = [row['response_id'] for row in human_result]

        # Filter out the AI ideas that match the human ideas and get their IDs
        filtered_ai_rows = AI_IDEAS_DF[AI_IDEAS_DF['aut_item'] == item].query("response not in @human_rows")
        ai_sample = filtered_ai_rows.sample(n_ai_ideas)
        ai_rows = ai_sample['response'].tolist()
        ai_ids = ai_sample['response_id'].tolist()

        # Add source labels if necessary
        if to_label:
            human_rows = [row + ' <span style="color: #ffffff;">(Source: <strong>Human</strong>)</span>' for row in
                          human_rows]
            ai_rows = [row + ' <span style="color: #ffffff;">(Source: <strong>A.I</strong>)</span>' for row in ai_rows]

        # Concatenate the human and AI rows and their IDs
        rows = human_rows + ai_rows
        ids = human_ids + ai_ids

        # Combine the rows and their IDs, shuffle them, and then separate them again
        rows_with_ids = list(zip(rows, ids))
        random.shuffle(rows_with_ids)
        rows, ids = zip(*rows_with_ids)
        data = zip(rows, ids)
        init_array = ','.join(map(str, ids))
        session['last_human_response'] = human_rows[-1]
        session.modified = True  # Explicitly mark the session as modified

        return render_template('render_trial.html', item=item, data=data, condition_no=condition_no,
                               label=SOURCE_LABEL if to_label else "", trial_no=condition_no + 1, init_array=init_array)


    # If the HTTP method is POST, insert the participant's response into the BigQuery table
    # then increment the condition_no and redirect to the next render_trial
    elif request.method == 'POST':

        # Retrieve the participant's response
        init_array = request.form.get('init_array', '').split(',')
        ranked_array = request.form.get('ranked_array', '').split(',')
        print("Init array and ranked array", init_array, ranked_array)
        response_text = request.form.get('participant_response')
        response_id = str(uuid.uuid4())
        session['responses'].append(response_text)
        session['response_ids'].append(response_id)
        session.modified = True  # Explicitly mark the session as modified
        response_similarity = calculate_similarity(response_text, session['last_human_response'])
        flash(f'Similarity Score: {response_similarity:.2f}')

        # Insert the participant's response into the BigQuery table
        row = {"item": item, "response_id": response_id, "participant_id": participant_id,
               "condition_order": condition_no, "response_text": response_text,
               "response_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "condition": condition,
               "world": session['world'], "init_array": init_array, "ranked_array": ranked_array, 'is_test':is_test}
        errors = client.insert_rows_json(table, [row])
        if not errors:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))

        # Redirect to next condition_no
        return redirect(url_for('render_trial', condition_no=condition_no + 1, method="GET"))


@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity_route():
    response_text = request.form.get('response')
    last_human_response = session.get('last_human_response')
    similarity_score = calculate_similarity(response_text, last_human_response)
    return str(max(round(similarity_score * 100), 1))


@app.route('/calculate_rank_similarity', methods=['POST'])
def calculate_rank_similarity_route():
    try:
        ranked_array = request.form.get('ranked_array', '').split(',')
        ranked_array_str = ",".join([f'"{item}"' for item in ranked_array])
        query = f"""
                SELECT response_id, IFNULL(rating, 2.5) as rating
                FROM `net_expr.responses`
                WHERE response_id IN UNNEST([{ranked_array_str}])
                """
        query_job = client.query(query)
        scores_dict = {row['response_id']: row['rating'] for row in query_job.result()}
        ranked_scores = [scores_dict.get(id, 2.5) for id in ranked_array]
        rank_similarity, _ = spearmanr(np.arange(1, len(ranked_array) + 1), ranked_scores)
        rank_similarity = (-1 * rank_similarity + 1) / 2  # -1 bc first idea is most creative
        return str(int(rank_similarity * 100))
    except Exception as e:
        print("Error:", e)
        random_value = random.uniform(0.3, 0.7)
        return str(int(random_value * 100))


@app.route("/reset_session")
def reset_session():
    session.clear()
    return "Session reset!"


@app.route("/get-graphs")
def get_graphs():
    """Generate graphs and return them as JSON"""
    participant_responses = list(zip(session['item_order'], session['responses']))
    participant_conditions = session['condition_order']

    # Get graphgs of responses
    human_graph, ai_graph, human_ai_graph, scores = make_graphs(participant_responses, participant_conditions,
                                                                file_prefix)

    # Add scores to database
    response_table = dataset.table("responses")
    rows_to_insert = [{"response_id": session['response_ids'][i], "rating": scores[i]} for i in range(len(scores))]
    errors = client.insert_rows_json(response_table, rows_to_insert)
    if errors:
        print(f"Encountered errors while inserting rows: {errors}")
    else:
        print("All rows have been added to responses.")
    return json.dumps({'human_graph': human_graph, 'ai_graph': ai_graph, 'human_ai_graph': human_ai_graph})


# These two functions are how we generate reports for specific participants.
# 1. The `results' endpoint calls the `thank_you.html` template passing in the UUID
# 2. From there, the `thank_you.html` template calls the `get_graphs_for_uuid` endpoint
@app.route("/results/<uuid>")
def results(uuid):
    """Results page for a specific UUID"""
    from_uuid_str = request.args.get('from_uuid', default='True')
    from_uuid = from_uuid_str.lower() == 'true'
    print(from_uuid)
    return render_template('thank_you.html', uuid=uuid, from_uuid=from_uuid)


@app.route("/get-graphs/<uuid>")
def get_graphs_for_uuid(uuid):
    """Generate graphs for a specific UUID and return them as JSON"""

    # Fetch the participant data from the database using the UUID
    # This is a simplified example; adjust this code to fit your actual database structure and API

    query = f"""
        SELECT responses.rating, trials.condition
        FROM `net_expr.trials` AS trials
        INNER JOIN `net_expr.responses` AS responses
        ON trials.response_id = responses.response_id
        WHERE trials.participant_id = '{uuid}'
        ORDER BY trials.world DESC
        LIMIT 5
    """
    query_job = client.query(query)
    results = list(query_job.result())  # Store the rows in a list

    # Convert the result into lists
    ratings = [row.rating for row in results]
    conditions = [row.condition for row in results]

    human_graph, ai_graph, human_ai_graph, _ = make_graphs(participant_responses=None,
                                                           conditions=conditions, participant_scores=ratings,
                                                           file_prefix=file_prefix)
    return json.dumps({'human_graph': human_graph, 'ai_graph': ai_graph, 'human_ai_graph': human_ai_graph})


@app.route('/feedback')
def feedback():
    return render_template('feedback_form.html')


@app.route('/submit-feedback-experiment', methods=['POST'])
def submit_feedback_experiment():
    if request.method == 'POST':
        experiment_feedback = request.form.get('experimentFeedback')
        experiment_feedback = experiment_feedback if experiment_feedback != '' else None

        # In the default case, the person came here after seeing resuls,
        # so we must have their UUID. But, let's say somebody shared resuls,
        # then random person decided to leave feedback -- we'd have no UUID.
        try:
            uuid = session['participant_id']
        except:
            uuid = None

        feedback_table = dataset.table("feedback")
        rows_to_insert = [{"feedback": experiment_feedback, 'participant_id': None, 'dt': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}]
        errors = client.insert_rows_json(feedback_table, rows_to_insert)
        if errors:
            print(f"Encountered errors while inserting rows: {errors}")
        else:
            print("All rows have been added to feedback.")
        return jsonify({'success': True})


def get_world():
    """Get the current world number.

    The reason why I get the minimum number of trials for each condition/item combination is that participants may
    have only completed half of the experiment, so there is not necessarily going to be an equal number of trials for
    everything.
    """
    query = """
        SELECT MIN(count) as min_count
        FROM (
          SELECT condition, item, COUNT(*) as count
          FROM `net_expr.trials`
          WHERE participant_id != "seed"
          GROUP BY condition, item
        )
    """
    query_job = client.query(query)
    results = query_job.result()
    trials = list(results)[0]['min_count']
    if not trials:
        trials = 0
    current_world = trials // N_PER_WORLD
    return current_world


if __name__ == '__main__':
    if is_local:
        app.run(port=5048, debug=True)
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(host="0.0.0.0", port=port)
