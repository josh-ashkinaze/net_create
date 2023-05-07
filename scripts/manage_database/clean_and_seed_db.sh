#!/bin/bash

python process_ai_responses.py
python get_human_seed_responses.py
python clear_table.py
python seed_db_with_human_responses.py