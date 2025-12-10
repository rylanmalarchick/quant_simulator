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
from src.backtest import walkforward
from src.backtest import walkforward_v2

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
    crisis_detector.save_crisis_cache(result)
    print(result['report'])


@cli.command('walkforward')
@click.option('--start', default=None, help='Start date (YYYY-MM-DD). Default: 1 year ago')
@click.option('--end', default=None, help='End date (YYYY-MM-DD). Default: today')
@click.option('--capital', default=100000.0, help='Initial capital')
@click.option('--train-window', default=252, help='Training window in days (default: 252 = 1 year)')
@click.option('--retrain-freq', default=21, help='Retrain frequency in days (default: 21 = monthly)')
@click.option('--top-k', default=5, help='Max positions to hold (default: 5)')
@click.option('--max-position', default=0.10, help='Max position size as fraction of portfolio (default: 0.10)')
@click.option('--slippage', default=10.0, help='Slippage in basis points (default: 10)')
@click.option('--serve', is_flag=True, help='Open report in browser when done')
def walkforward_cmd(start, end, capital, train_window, retrain_freq, top_k, max_position, slippage, serve):
    """
    Run walk-forward backtesting on your trading strategy.
    
    This performs proper out-of-sample validation with:
    - No look-ahead bias (features calculated with only past data)
    - Periodic model retraining (expanding window)
    - Realistic execution costs (slippage + commissions)
    - Benchmark comparison vs SPY
    - Statistical significance testing
    
    Example:
        python -m src.cli walkforward --start 2023-01-01 --end 2024-12-01 --serve
    """
    import os
    import webbrowser
    
    # Default dates
    if end is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end, '%Y-%m-%d')
    
    if start is None:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = datetime.strptime(start, '%Y-%m-%d')
    
    # Load tickers from config
    tickers_config = load_tickers()
    tickers = tickers_config.get('quantum', [])
    
    if not tickers:
        print("ERROR: No tickers found in tickers.yml under 'quantum' category")
        sys.exit(1)
    
    print("=" * 70)
    print("                   WALK-FORWARD BACKTEST")
    print("=" * 70)
    print(f"Period:          {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${capital:,.2f}")
    print(f"Tickers:         {len(tickers)} symbols")
    print(f"Train Window:    {train_window} days")
    print(f"Retrain Freq:    {retrain_freq} days")
    print(f"Max Positions:   {top_k}")
    print(f"Max Position %:  {max_position:.1%}")
    print(f"Slippage:        {slippage} bps")
    print("=" * 70)
    print()
    print("Fetching data and running backtest...")
    print("(This may take several minutes)")
    print()
    
    try:
        # Run the backtest
        backtester = walkforward.WalkForwardBacktester(
            tickers=tickers,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_capital=capital,
            train_window_days=train_window,
            retrain_frequency=retrain_freq,
            top_k=top_k,
            max_position_pct=max_position,
            slippage_bps=slippage,
        )
        result = backtester.run()
        
        # Generate report
        report_path = walkforward.generate_backtest_report(result)
        abs_path = os.path.abspath(report_path)
        
        # Print summary
        print(result.summary())
        
        print(f"Full report saved to: {abs_path}")
        
        if serve:
            print("Opening report in browser...")
            webbrowser.open(f"file://{abs_path}")
        
    except Exception as e:
        print(f"ERROR: Backtest failed - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command('walkforward-v2')
@click.option('--start', default=None, help='Start date (YYYY-MM-DD). Default: 2 years ago')
@click.option('--end', default=None, help='End date (YYYY-MM-DD). Default: today')
@click.option('--capital', default=100000.0, help='Initial capital')
@click.option('--train-window', default=252, help='Training window in days (default: 252)')
@click.option('--retrain-freq', default=21, help='Retrain frequency in days (default: 21)')
@click.option('--top-k', default=5, help='Max positions to hold (default: 5)')
@click.option('--max-position', default=0.15, help='Max position size (default: 0.15)')
@click.option('--slippage', default=10.0, help='Slippage in basis points (default: 10)')
@click.option('--forward-days', default=5, help='Prediction horizon in days (default: 5)')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration')
@click.option('--serve', is_flag=True, help='Open report in browser when done')
def walkforward_v2_cmd(start, end, capital, train_window, retrain_freq, top_k, max_position, slippage, forward_days, no_gpu, serve):
    """
    Run ENHANCED walk-forward backtesting (V2).
    
    Improvements over V1:
    - 5-day risk-adjusted return targets (not binary)
    - 50+ technical indicators with Numba JIT
    - Market regime detection (bull/bear/sideways)
    - Conviction-weighted position sizing
    - GPU acceleration via XGBoost CUDA
    - Parallel processing (28 CPU threads)
    
    Example:
        python -m src.cli walkforward-v2 --start 2023-01-01 --end 2025-12-01 --serve
    """
    import os
    import webbrowser
    
    # Default dates
    if end is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end, '%Y-%m-%d')
    
    if start is None:
        start_date = end_date - timedelta(days=730)  # 2 years default
    else:
        start_date = datetime.strptime(start, '%Y-%m-%d')
    
    # Load tickers
    tickers_config = load_tickers()
    tickers = tickers_config.get('quantum', [])
    
    if not tickers:
        print("ERROR: No tickers found in tickers.yml under 'quantum' category")
        sys.exit(1)
    
    use_gpu = not no_gpu
    
    print("=" * 70)
    print("            WALK-FORWARD BACKTEST V2 (ENHANCED)")
    print("=" * 70)
    print(f"Period:           {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Capital:  ${capital:,.2f}")
    print(f"Tickers:          {len(tickers)} symbols")
    print(f"Train Window:     {train_window} days")
    print(f"Retrain Freq:     {retrain_freq} days")
    print(f"Forward Horizon:  {forward_days} days (risk-adjusted target)")
    print(f"Max Positions:    {top_k}")
    print(f"Max Position %:   {max_position:.1%}")
    print(f"Slippage:         {slippage} bps")
    print("-" * 70)
    print(f"GPU Acceleration: {'ENABLED (XGBoost CUDA)' if use_gpu else 'DISABLED'}")
    print(f"CPU Threads:      {walkforward_v2.N_JOBS}")
    print(f"Features:         50+ indicators (Numba JIT compiled)")
    print(f"Regime Detection: Enabled (bull/bear/sideways)")
    print("=" * 70)
    print()
    print("Running backtest... (this will utilize your hardware)")
    print()
    
    try:
        import time
        start_time = time.time()
        
        backtester = walkforward_v2.WalkForwardBacktesterV2(
            tickers=tickers,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_capital=capital,
            train_window_days=train_window,
            retrain_frequency=retrain_freq,
            top_k=top_k,
            max_position_pct=max_position,
            slippage_bps=slippage,
            forward_days=forward_days,
            use_gpu=use_gpu,
        )
        result = backtester.run()
        
        elapsed = time.time() - start_time
        
        # Generate report
        report_path = walkforward_v2.generate_backtest_report_v2(result)
        abs_path = os.path.abspath(report_path)
        
        # Print summary
        print(result.summary())
        
        print(f"Execution time: {elapsed:.1f} seconds")
        print(f"Full report saved to: {abs_path}")
        
        if serve:
            print("Opening report in browser...")
            webbrowser.open(f"file://{abs_path}")
        
    except Exception as e:
        print(f"ERROR: Backtest failed - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    cli()
