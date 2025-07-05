import click
from src.backtest.engine import run_backtest

@click.command()
@click.option('--start', required=True, type=str, help='Start date for the backtest (YYYY-MM-DD)')
@click.option('--end', required=True, type=str, help='End date for the backtest (YYYY-MM-DD)')
def backtest(start, end):
    """
    Runs a backtest of the trading strategy.
    """
    print(f"Running backtest from {start} to {end}...")
    run_backtest(start, end)

if __name__ == '__main__':
    backtest()
