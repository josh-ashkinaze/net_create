#!/bin/bash

# Get the current date and time
NOW=$(date +"%Y_%m_%d_%H%M%S")
LOGFILE="${NOW}_data_process_pipeline.log"

echo "Pulling data from the database" | tee -a $LOGFILE
jupyter nbconvert --execute get_data.ipynb >> $LOGFILE 2>&1

echo "Adding elaboration features" | tee -a $LOGFILE
python3 add_elaboration_feats.py >> $LOGFILE 2>&1

#echo "Adding embedding features" | tee -a $LOGFILE
#python3 add_embedding_feats.py >> $LOGFILE 2>&1

echo "Done" | tee -a $LOGFILE
