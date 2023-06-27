#!/bin/bash

# optional: clear_all_experiment_data.py --remove_seeds
python clear_all_experiment_data.py
python process_ai_responses.py
python score_ai_responses.py
python get_human_seed_responses.py
python seed_db_with_human_responses.py
