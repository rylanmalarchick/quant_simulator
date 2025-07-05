# quantSim

A quantitative trading bot that uses machine learning to generate trading signals and execute them in a paper trading account. This bot operates on a **daily** timeframe, making it a **swing-trading** system.

## Overview

This project is an end-to-end quantitative trading system designed to be run in a containerized environment using Docker. The system performs the following steps:

1.  **Data Ingestion**: Ingests **daily** historical market data from **Yahoo Finance** (using the `yfinance` library) and stores it in a local SQLite database. It also attempts to ingest alternative data on Pelosi trades, though this source is currently unreliable.
2.  **Feature Engineering & Model Training**: Creates features from the daily data (e.g., returns, volatility) and trains a LightGBM model to predict the probability of the next day's price movement.
3.  **Signal Generation**: Generates BUY/SELL signals based on the model's daily predictions.
4.  **Trade Execution**: Executes trades in an Alpaca paper trading account based on the daily signals.
5.  **Backtesting**: Provides a backtesting engine to simulate and evaluate strategy performance on historical data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- A Linux-based OS with `cron` and `anacron`.
- Python 3.10+
- Docker and Docker Compose (Optional)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rylanmalarchick/quantSim.git
    cd quantSim
    ```

2.  **Set up the environment:**
    - Create and activate a Python virtual environment:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    - Install the dependencies:
      ```bash
      pip install -r requirements.txt
      ```

3.  **Configure the application:**
    - In the `.qtick` directory, you should have a `config.yaml` file with your API key for Alpaca. The Finnhub key is no longer used.
    - Review the `tickers.yml` file to customize the initial universe of stocks.

## Automated Trading with Cron

The most effective way to run this bot is to set it up on a schedule. A setup script is provided to configure `anacron` and `cron` to run the pipeline automatically.

**How it Works:**
- **`anacron`** will run the pre-market tasks (`ingest`, `features`, `train_model`) once a day. Because it's `anacron`, if your computer is off at the scheduled time, it will run the tasks as soon as you boot up.
- **`cron`** will run the trading tasks (`signal` and `execute`) once a day at a fixed time during market hours on weekdays.

### Setup

To configure the schedule, run the provided setup script. You will be prompted for your password as it needs to modify a system file (`/etc/anacrontab`).

```bash
chmod +x setup_cron.sh
./setup_cron.sh
```

This will:
1.  Create wrapper scripts in the `scripts/` directory.
2.  Add a daily job to `/etc/anacrontab` for pre-market data ingestion and model training.
3.  Add a daily job to your user's `crontab` to run the trading logic.
4.  Create a log file at `logs/cron.log` where you can monitor the output of the scheduled tasks.

## Manual Execution

You can also run the pipeline manually for testing or debugging.

### 1. Full Pipeline Execution

A master script, `src/run.py`, orchestrates the entire pipeline.

- **Locally:**
  ```bash
  python -m src.run
  ```

- **With Docker:**
  ```bash
  docker-compose build
  docker-compose run --rm pipeline
  ```

### 2. Step-by-Step Execution

For more granular control, run each step individually using the CLI.

- **Locally:**
  ```bash
  python -m src.cli ingest
  python -m src.cli features
  python -m src.cli train_model
  python -m src.cli signal
  python -m src.cli execute
  ```

- **With Docker:**
  ```bash
  docker-compose run --rm cli ingest
  # etc.
  ```
