#!/bin/bash
# This script runs the intraday trading tasks: signal generation and execution.

# Navigate to the project directory
cd /home/rylan/Documents/school/projects/quantSim || exit

# Activate the virtual environment
source venv/bin/activate

# Run the commands
echo "--- Generating trading signals ---"
python -m src.cli signal

echo "--- Executing trades ---"
python -m src.cli execute

echo "--- Generating dashboard ---"
python -m src.cli dashboard

echo "--- Intraday trading cycle complete ---"
