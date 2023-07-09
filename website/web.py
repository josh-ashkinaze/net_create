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
import itertools
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
from helpers.helpers import catch_if_none, insert_into_bigquery, do_sql_query, get_participant_data

############################################################################################################
# Environment variables
############################################################################################################
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
app.secret_key = flask_secret_key

if is_local:
    is_test = True
else:
    is_test = True if os.environ.get('IS_TEST') == "True" else False

# Connect to BQ
if not is_local:
    json_key = json.loads(os.environ['GOOGLE_CREDS'])
    credentials = service_account.Credentials.from_service_account_info(json_key)
else:
    key_path = f"{file_prefix}secrets/google_creds.json"
    credentials = service_account.Credentials.from_service_account_file(key_path,
                                                                        scopes=[
                                                                            "https://www.googleapis.com/auth/cloud-platform"], )
############################################################################################################
############################################################################################################

############################################################################################################
# Parameters
############################################################################################################
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
PROFANE_WORDS = set(w.lower() for w in pd.read_csv(file_prefix + "helpers/profane_words.txt", header=None)[0].tolist())

############################################################################################################
############################################################################################################


# START THE EXPERIMENT
############################################################################################################
@app.route("/")
def consent_form():
    """Consent form"""
    session['request_args'] = catch_if_none(request.query_string.decode(), "string")
    session['referer'] = catch_if_none(request.headers.get('Referer'), "string")

    if request.query_string.decode() == "from=prolific":
        session['is_prolific'] = True
    else:
        session['is_prolific'] = False

    if is_local or os.environ.get('IS_TEST') == "True" or request.query_string.decode()=="how=test" or catch_if_none(request.headers.get('Referer'), "string")=='https://dashboard.heroku.com/':
        session['test'] = True
    else:
        session['test'] = False
    session.modified = True
    return render_template('consent_form.html', is_prolific=session['is_prolific'])


@app.route("/start-experiment")
def start_experiment():
    """
    Start the experiment.

    1. Create a UUID for participant ID
    2. Get stuff particant submitted add to DB
    3. Counterbalance conditions and counterbalance items.
    4. Redirect to the first trial.
    """

    # Assign UUID and world
    session['participant_id'] = str(uuid.uuid4())

    item_order, condition_order = get_lowest_sum_subset(client)
    session['item_order'] = item_order
    session['condition_order'] = condition_order
    print("After randomization, item order is: ", item_order)
    print("After randomization, condition order is: ", condition_order)
    session.modified = True
    # Init lists of responses
    session['responses'] = []
    session['response_ids'] = []
    session.modified = True

    # Get args and insert into db
    creativity_ai = request.args.get('creativitySliderAIValue')
    creativity_human = request.args.get('creativitySliderHumanValue')
    ai_feeling = request.args.get('aiFeelingValue')
    country = request.args.get('countryValue')
    age = request.args.get('ageValue')
    gender = request.args.get('genderValue')
    gender_describe = request.args.get('gender-describeValue')
    prolific_id = request.args.get("prolific_idValue")
    row = {'participant_id': session['participant_id'],
           'dt': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
           'creativity_ai': catch_if_none(creativity_ai, "number"),
           'creativity_human': catch_if_none(creativity_human, "number"),
           'age': catch_if_none(age, "number"),
           'ai_feeling': catch_if_none(ai_feeling, "string"),
           'country': catch_if_none(country, "string"),
           'prolific_id': catch_if_none(prolific_id, "string"),
           'is_prolific': session['is_prolific'],
           'request_args': catch_if_none(session['request_args'], "string"),
           'referer': catch_if_none(session['referer'], "string"),
           'gender':catch_if_none(gender, "string"),
           'gender_describe': catch_if_none(gender_describe, "string"),
           'is_test': session['test']
           }
    print(row)
    person_table = dataset.table("person")
    insert_into_bigquery(client, person_table, [row])
    return redirect(url_for('render_trial', condition_no=0, method="GET"))


@app.route("/render_trial/<int:condition_no>", methods=['GET', 'POST'])
def render_trial(condition_no):
    """
    Recursively handles the render_trial route for a particular condition_no.

    The idea is that in a session object (like a unique dictionary or each participant),
    we store the participant ID, the condition order, and the item order.
    Then, we keep calling this function with the next condition_no -- which indexes the participant's items and condition sequence --
    until we've gone through all trials. After trials, return thank you page.

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
    if condition_no > len(ITEMS) - 1:
        return redirect(url_for('results', uuid=session['participant_id']) + f"?from_uuid=False&is_prolific={session['is_prolific']}")

    participant_id = session['participant_id']
    condition = session['condition_order'][condition_no]
    to_label = CONDITIONS[condition]['label']
    n_human_ideas = CONDITIONS[condition]['n_human']
    n_ai_ideas = CONDITIONS[condition]['n_ai']
    item = session['item_order'][condition_no]

    if request.method == "GET":
        # COUNT(*)+1 because we want to make sure we don't go over N_PER_WORLD.
        human_query = f"""
            WITH current_world AS (
    SELECT 
        FLOOR((COUNT(*)+1 )/ {N_PER_WORLD}) as world_value 
    FROM 
        `net_expr.trials`
    WHERE 
        is_test IS FALSE
        AND participant_id != 'seed'
        AND item = '{item}'
        AND condition = '{condition}'
)

SELECT 
    response_text, 
    response_id, 
    current_world.world_value as world
FROM (
    SELECT 
        response_text, 
        response_id, 
        world, 
        response_date
    FROM 
        `net_expr.trials`
    WHERE 
        item = '{item}' 
        AND condition = '{condition}' 
        AND is_test = False 
        AND is_profane is not TRUE
) AS subquery
JOIN
    current_world
ON 
    subquery.world = current_world.world_value
ORDER BY 
    subquery.response_date DESC
LIMIT 
    {n_human_ideas}

         """
        # print(human_query)
        human_result = do_sql_query(client, human_query)
        print(human_result)
        human_rows = [row['response_text'] for row in human_result]
        human_ids = [row['response_id'] for row in human_result]
        session['world'] = human_result[0]['world']
        session.modified = True

        filtered_ai_rows = AI_IDEAS_DF[AI_IDEAS_DF['aut_item'] == item].query("response not in @human_rows")
        ai_sample = filtered_ai_rows.sample(n_ai_ideas)
        ai_rows = ai_sample['response'].tolist()
        ai_ids = ai_sample['response_id'].tolist()

        if to_label:
            human_rows = [f"{row} <span style='color: #ffffff;'>(Source: <strong>Human</strong>)</span>" for row in
                          human_rows]
            ai_rows = [f"{row} <span style='color: #ffffff;'>(Source: <strong>A.I</strong>)</span>" for row in ai_rows]

        # Combine the rows and their IDs, shuffle them, and then separate them again
        rows_with_ids = list(zip(human_rows + ai_rows, human_ids + ai_ids))
        random.shuffle(rows_with_ids)
        rows, ids = zip(*rows_with_ids)
        init_array = ','.join(map(str, ids))

        session['last_human_response'] = human_rows[-1]
        session.modified = True

        return render_template('render_trial.html', item=item, data=zip(rows, ids), condition_no=condition_no,
                               label=SOURCE_LABEL if to_label else "", trial_no=condition_no + 1, init_array=init_array,
                               is_prolific=session['is_prolific'])


    # If the HTTP method is POST, insert the participant's response into the BigQuery table
    # then increment the condition_no and redirect to the next render_trial
    elif request.method == 'POST':

        # Retrieve the participant's response
        init_array = request.form.get('init_array', '').split(',')
        ranked_array = request.form.get('ranked_array', '').split(',')
        response_text = request.form.get('participant_response')
        duration = catch_if_none(request.form.get('duration'), "float")
        response_id = str(uuid.uuid4())
        session['responses'].append(response_text)
        session['response_ids'].append(response_id)
        session.modified = True  # Explicitly mark the session as modified
        # Insert the participant's response into the BigQuery table
        row = {"item": item, "response_id": response_id, "participant_id": participant_id,
               "condition_order": condition_no, "response_text": response_text,
               "response_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "condition": condition,
               "world": session['world'], "init_array": init_array, "ranked_array": ranked_array, 'is_test': session['test'],
               "duration": duration, 'is_profane':contains_profanity(response_text, PROFANE_WORDS)}
        print(row)
        errors = insert_into_bigquery(client, table, [row])
        # Redirect to next condition_no
        return redirect(url_for('render_trial', condition_no=condition_no + 1, method="GET"))


@app.route('/get-duration', methods=['POST'])
def get_duration():
    duration = request.form.get('duration')
    try:
        if not duration:
            return None
        else:
            return str(np.round(duration, 4))
    except Exception as e:
        print(e)
        return str(-1)


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
    print(participant_responses)

    # Get graphgs of responses
    human_graph, ai_graph, human_ai_graph, scores = make_graphs(participant_responses, participant_conditions,
                                                                file_prefix)

    # Add scores to database
    response_table = dataset.table("responses")
    rows_to_insert = [{"response_id": session['response_ids'][i], "rating": scores[i]} for i in range(len(scores))]
    insert_into_bigquery(client, response_table, rows_to_insert)
    return json.dumps({'human_graph': human_graph, 'ai_graph': ai_graph, 'human_ai_graph': human_ai_graph})


# These two functions are how we generate reports for specific participants.
# 1. The `results' endpoint calls the `thank_you.html` template passing in the UUID
# 2. From there, the `thank_you.html` template calls the `get_graphs_for_uuid` endpoint
@app.route("/results/<uuid>")
def results(uuid):
    """Results page for a specific UUID"""
    from_uuid_str = request.args.get('from_uuid', default='True')
    from_uuid = from_uuid_str.lower() == 'true'
    is_prolific_str = request.args.get('is_prolific', default='False')
    is_prolific = is_prolific_str.lower() == 'true'
    return render_template('thank_you.html',
                           uuid=uuid,
                           from_uuid=from_uuid,
                           request_args=request.query_string.decode().replace("from_uuid=False&", ""),
                           is_prolific=is_prolific)


@app.route("/get-graphs/<uuid>")
def get_graphs_for_uuid(uuid):
    """Generate graphs for a specific UUID and return them as JSON"""

    # Fetch the participant data from the database using the UUID
    # This is a simplified example; adjust this code to fit your actual database structure and API

    results = get_participant_data(client, uuid)
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
        experiment_feedback = catch_if_none(request.form.get('experimentFeedback'), "string")
        ai_thoughts = catch_if_none(request.form.get('aiThoughts'), "string")  # get the 'aiThoughts' field

        # In the default case, the person came here after seeing results,
        # so we must have their UUID. But, let's say somebody shared results,
        # then random person decided to leave feedback -- we'd have no UUID.
        try:
            pid = session['participant_id']
        except:
            pid = str(uuid.uuid4())

        feedback_table = dataset.table("feedback")
        rows_to_insert = [{"feedback": experiment_feedback,
                           'participant_id': pid,
                           'dt': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                           'ai_thoughts': ai_thoughts}]  # include 'ai_thoughts' in the row data
        errors = insert_into_bigquery(client, feedback_table, rows_to_insert)
        return jsonify({'success': True})


@app.route('/record_duration_route', methods=['POST'])
def record_duration_route():
    duration = request.form.get('duration')

    # Process the duration as needed
    # For example, you can save it to the database, etc.
    print("DURATION", duration)
    return jsonify({"message": "Duration recorded successfully", "duration": duration})



def get_lowest_sum_subset(client):
    # Execute the SQL query to get the unique `item` and `condition` pairs along with their counts.
    query = f"""
        SELECT
            item,
            condition,
            COUNT(*) as count
        FROM
            `net_expr.trials`
        WHERE
            NOT is_test
        GROUP BY
            item, condition
    """
    query_job = client.query(query)
    result = query_job.result()

    # Convert the BigQuery result to a Pandas DataFrame.
    data = []
    for row in result:
        data.append({"item": row["item"], "condition": row["condition"], "count": row["count"]})
    df = pd.DataFrame(data)

    # Get all possible orderings of items and conditions.
    unique_items = df["item"].unique()
    unique_conditions = df["condition"].unique()

    # Create a dictionary to store the counts for each (item, condition) pair.
    count_dict = {(row["item"], row["condition"]): int(row["count"]) for _, row in df.iterrows()}

    item_permutations = list(itertools.permutations(unique_items, len(unique_items)))
    condition_permutations = list(itertools.permutations(unique_conditions, len(unique_conditions)))

    # Take the product of item and condition permutations.
    item_condition_permutations = list(itertools.product(item_permutations, condition_permutations))

    min_sum = float("inf")
    lowest_sum_subset = None

    for item_perm, condition_perm in item_condition_permutations:
        subset = list(zip(item_perm, condition_perm))
        total_sum = sum(count_dict[item_condition] for item_condition in subset)

        if total_sum < min_sum:
            min_sum = total_sum
            lowest_sum_subset = subset

    items = [x[0] for x in lowest_sum_subset]
    conditions = [x[1] for x in lowest_sum_subset]
    return items, conditions


def contains_profanity(input_string, prof_words_set):
    input_string = input_string.lower()
    words = input_string.split()
    for word in words:
        if word in prof_words_set:
            return True
    return False

if __name__ == '__main__':
    if is_local:
        app.run(port=5048, debug=True)
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(host="0.0.0.0", port=port)
