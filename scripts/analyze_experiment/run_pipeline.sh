#!/bin/bash

# Get the current date and time
NOW=$(date +"%Y_%m_%d_%H%M%S")
LOGFILE="${NOW}_data_process_pipeline.log"

echo "Pulling data from the database" | tee -a $LOGFILE
papermill get_data.ipynb get_data.ipynb

echo "Adding elaboration features" | tee -a $LOGFILE
python3 add_elaboration_feats.py >> $LOGFILE 2>&1

echo "Adding originality features" | tee -a $LOGFILE
#python3 add_originality_feats.py >> $LOGFILE 2>&1

echo "Adding diversity analysis" | tee -a $LOGFILE
papermill diversity_metrics.ipynb diversity_metrics.ipynb

echo "Running R notebook"
Rscript -e "rmarkdown::render('r_models.Rmd', output_file='r_models.html')" >> $LOGFILE 2>&1

echo "Done" | tee -a $LOGFILE
