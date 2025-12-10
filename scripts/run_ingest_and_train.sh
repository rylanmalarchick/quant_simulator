#!/bin/bash
# This script runs the daily pre-market tasks: data ingestion, model training, and crisis check.

# Navigate to the project directory
cd /home/rylan/Documents/school/projects/quantSim || exit

# Activate the virtual environment
source venv/bin/activate

# Run the commands
echo "--- Starting daily data ingestion ---"
python -m src.cli ingest

echo "--- Starting daily model training ---"
python -m src.cli train_model

echo "--- Running daily crisis check ---"
python -m src.cli crisis

echo "--- Pre-market tasks complete ---"
