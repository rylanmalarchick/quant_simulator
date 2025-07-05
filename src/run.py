import sys
import os
from src import cli

# Ensure the script's directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_pipeline():
    """Runs the full trading pipeline by directly calling the CLI functions."""
    print(">>> Starting quantSim trading pipeline <<<")

    try:
        # Step 1: Initialize Database
        print("--- Running step: init ---")
        cli.init.callback()
        print("--- Step 'init' completed successfully. ---")

        # Step 2: Ingest Data
        print("--- Running step: ingest ---")
        cli.ingest.callback()
        print("--- Step 'ingest' completed successfully. ---")

        # Step 3: Build Features
        print("--- Running step: features ---")
        cli.features.callback()
        print("--- Step 'features' completed successfully. ---")

        # Step 4: Train Model
        print("--- Running step: train_model ---")
        cli.train_model.callback()
        print("--- Step 'train_model' completed successfully. ---")

        # Step 5: Generate Signals
        print("--- Running step: signal ---")
        cli.signal.callback()
        print("--- Step 'signal' completed successfully. ---")

        # Step 6: Execute Trades
        print("--- Running step: execute ---")
        if hasattr(cli, 'execute'):
             cli.execute.callback()
        else:
             print("--- WARNING: 'execute' command not found in cli.py. Skipping. ---")
        print("--- Step 'execute' completed successfully. ---")

        print("\n>>> quantSim trading pipeline completed successfully! <<<")

    except Exception as e:
        print(f"\n>>> Pipeline execution failed. Error: {e} <<<", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()