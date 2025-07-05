#!/bin/bash

# This script sets up the anacron and cron jobs for the quantSim trading bot.

# --- Configuration ---
PROJECT_DIR="/home/rylan/Documents/school/projects/quantSim"
LOG_FILE="$PROJECT_DIR/logs/cron.log"
ANACRON_CONF="/etc/anacrontab"
PRE_MARKET_SCRIPT="$PROJECT_DIR/scripts/run_ingest_and_train.sh"
TRADING_SCRIPT="$PROJECT_DIR/scripts/run_signal_and_execute.sh"

# --- Setup ---

# Create log directory and file
mkdir -p "$PROJECT_DIR/logs"
touch "$LOG_FILE"

echo "--- Setting up cron and anacron jobs for quantSim ---"

# --- Anacron Job for Pre-Market Tasks ---

# Job definition for anacron
ANACRON_JOB="1	5	quantSim.premarket	/bin/bash $PRE_MARKET_SCRIPT >> $LOG_FILE 2>&1"

# Check if the job already exists in anacrontab
if grep -q "quantSim.premarket" "$ANACRON_CONF"; then
    echo "Anacron job 'quantSim.premarket' already exists. Skipping."
else
    echo "Adding 'quantSim.premarket' job to $ANACRON_CONF..."
    # Append the job to anacrontab. This requires sudo.
    echo -e "\n# quantSim pre-market job (ingestion and training)" | sudo tee -a "$ANACRON_CONF" > /dev/null
    echo "$ANACRON_JOB" | sudo tee -a "$ANACRON_CONF" > /dev/null
    echo "Anacron job added."
fi

# --- Cron Jobs for Intraday Trading ---

# Cron job definitions
CRON_JOB_1="30 9 * * 1-5 /bin/bash $TRADING_SCRIPT >> $LOG_FILE 2>&1"  # 9:30 AM on weekdays
CRON_JOB_2="0 11 * * 1-5 /bin/bash $TRADING_SCRIPT >> $LOG_FILE 2>&1"  # 11:00 AM on weekdays
CRON_JOB_3="30 12 * * 1-5 /bin/bash $TRADING_SCRIPT >> $LOG_FILE 2>&1" # 12:30 PM on weekdays
CRON_JOB_4="0 14 * * 1-5 /bin/bash $TRADING_SCRIPT >> $LOG_FILE 2>&1"  # 2:00 PM on weekdays

# Function to add a cron job if it doesn't exist
add_cron_job() {
    local job_to_add=$1
    (crontab -l 2>/dev/null | grep -Fq "$job_to_add") || (crontab -l 2>/dev/null; echo "$job_to_add") | crontab -
}

echo "Adding intraday trading jobs to user's crontab..."
add_cron_job "# quantSim intraday trading jobs"
add_cron_job "$CRON_JOB_1"
add_cron_job "$CRON_JOB_2"
add_cron_job "$CRON_JOB_3"
add_cron_job "$CRON_JOB_4"
echo "Cron jobs added."

echo "--- Setup complete. ---"
echo "Log file for scheduled tasks is at: $LOG_FILE"
