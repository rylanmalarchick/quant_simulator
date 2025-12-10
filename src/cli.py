import click
import yaml
import sys
# Use the new market_data_client
from src.ingest import sqlite_writer, market_data_client, nav_client
from src.train import train
from src.signal import generate
from src.backtest import cli as backtest_cli
from src.config import load_tickers
from datetime import datetime, timedelta

@click.group()
def cli():
    """
    A CLI for the quant trading system.
    """
    pass

@cli.command()
def init():
    """
    Initializes the database.
    """
    sqlite_writer.create_tables()
    print("Database initialized.")

@cli.command()
def ingest():
    """
    Ingests data from all sources.
    """
    print("Ingesting data...")

    # Load tickers from the configuration file
    tickers = load_tickers()
    all_tickers = tickers.get('quantum', []) + tickers.get('fidelity_mutual_funds', [])
    
    # Add market benchmarks for context features (SPY for market regime, VIX for volatility regime)
    # These are used in Tier 3 features but not traded directly
    market_benchmarks = ['SPY', '^VIX']
    all_tickers_with_benchmarks = list(set(all_tickers + market_benchmarks))

    # Define time range for data ingestion (last 365 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Ingest market data using yfinance
    print("Ingesting market data from Yahoo Finance...")
    ingested_tickers = 0
    for ticker in all_tickers_with_benchmarks:
        # yfinance expects date strings in 'YYYY-MM-DD' format
        print(f"Fetching data for {ticker}...")
        market_data = market_data_client.get_market_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if market_data and market_data.get('s') == 'ok':
            formatted_data = []
            for i in range(len(market_data['t'])):
                formatted_data.append((
                    ticker,
                    market_data['t'][i],
                    market_data['o'][i],
                    market_data['h'][i],
                    market_data['l'][i],
                    market_data['c'][i],
                    market_data['v'][i]
                ))
            if formatted_data:
                sqlite_writer.write_raw_bars(formatted_data)
                print(f"Data for {ticker} ingested.")
                ingested_tickers += 1
            else:
                print(f"No data to write for {ticker}.")
        else:
            print(f"Could not fetch or no data for {ticker}.")
    
    if ingested_tickers == 0:
        raise Exception("No market data was ingested. Halting pipeline.")

    print("Data ingestion complete.")

from src.features import builder

@cli.command()
def features():
    """
    Builds features for the model.
    """
    builder.build_and_persist_features()

@cli.command()
def train_model():
    """
    Trains the model.
    """
    train.train_model()

from src.signal import generate
from src.exec import execute as trading_executor
from src.dashboard import generator as dashboard_generator
from src.crisis import detector as crisis_detector

@cli.command()
def execute():
    """
    Executes trades based on the latest signals.
    """
    print("Executing trades...")
    trading_executor.execute_signals()
    print("Trade execution complete.")

@cli.command()
def signal():
    """
    Generates trading signals.
    """
    generate.generate_signals()

@cli.command()
@click.argument('symbol')
@click.option('--category', required=True, type=click.Choice(['quantum']))
def add_symbol(symbol, category):
    """
    Adds a symbol to the specified category in tickers.yml.
    """
    tickers = load_tickers()
    if symbol not in tickers[category]:
        tickers[category].append(symbol)
        with open('tickers.yml', 'w') as f:
            yaml.dump(tickers, f)
        print(f"Added {symbol} to {category}.")
    else:
        print(f"{symbol} already exists in {category}.")

@cli.command()
@click.argument('symbol')
@click.option('--category', required=True, type=click.Choice(['quantum']))
def remove_symbol(symbol, category):
    """
    Removes a symbol from the specified category in tickers.yml.
    """
    tickers = load_tickers()
    if symbol in tickers[category]:
        tickers[category].remove(symbol)
        with open('tickers.yml', 'w') as f:
            yaml.dump(tickers, f)
        print(f"Removed {symbol} from {category}.")
    else:
        print(f"{symbol} does not exist in {category}.")

cli.add_command(backtest_cli.backtest)

@cli.command()
@click.option('--serve', is_flag=True, help='Start a local HTTP server to view the dashboard')
@click.option('--port', default=8080, help='Port for the HTTP server')
def dashboard(serve, port):
    """
    Generates the HTML performance dashboard.
    """
    import os
    import webbrowser
    
    path = dashboard_generator.generate_dashboard()
    abs_path = os.path.abspath(path)
    
    print(f"Dashboard generated: {abs_path}")
    
    if serve:
        import http.server
        import socketserver
        import threading
        
        dashboard_dir = os.path.dirname(abs_path)
        os.chdir(dashboard_dir)
        
        handler = http.server.SimpleHTTPRequestHandler
        
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://localhost:{port}/index.html"
            print(f"Serving dashboard at: {url}")
            print("Press Ctrl+C to stop the server...")
            webbrowser.open(url)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")
    else:
        file_url = f"file://{abs_path}"
        print(f"Open in browser: {file_url}")
        webbrowser.open(file_url)


@cli.command()
def crisis():
    """
    Run the crisis/bubble detector using statistical mechanics indicators.
    
    Analyzes market conditions for signs of bubble/crisis using:
    - Cross-asset correlation (herd behavior)
    - VIX ratio (fear level)  
    - Return autocorrelation (critical slowing down)
    - Return kurtosis (fat tails)
    - Market breadth (participation)
    """
    print("Running crisis/bubble detector...")
    print("(This may take a moment to fetch market data)")
    print()
    
    result = crisis_detector.run_crisis_check()
    print(result['report'])


if __name__ == '__main__':
    cli()
