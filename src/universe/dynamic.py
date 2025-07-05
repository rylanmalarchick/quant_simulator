import pandas as pd
from src.ingest import market_data_client # Use the new client
from src.universe.loader import load_and_validate_universe
from datetime import datetime, timedelta
from src.logging import get_logger

logger = get_logger(__name__)

def calculate_dynamic_universe():
    """
    Calculates the dynamic universe of the top 40 performing stocks from the base universe.
    """
    logger.info("Calculating dynamic universe...")
    universe = load_and_validate_universe()
    
    # Use the quantum tickers as the base for the dynamic universe calculation
    base_tickers = universe.get('quantum', [])
    
    # Calculate 30-day returns
    returns = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for ticker in base_tickers:
        try:
            # Use the new market_data_client
            candles = market_data_client.get_market_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if candles and candles.get('s') == 'ok' and len(candles['c']) > 1:
                price_30_days_ago = candles['c'][0]
                current_price = candles['c'][-1]
                returns.append((ticker, (current_price - price_30_days_ago) / price_30_days_ago))
        except Exception as e:
            logger.error(f"Could not get data for {ticker} for dynamic universe calculation: {e}")

    # Select top 40 performers
    returns.sort(key=lambda x: x[1], reverse=True)
    dynamic_universe = [ticker for ticker, _ in returns[:40]]
    
    logger.info(f"Calculated dynamic universe with {len(dynamic_universe)} tickers.")
    return dynamic_universe

if __name__ == '__main__':
    dynamic_universe = calculate_dynamic_universe()
    print("Dynamic universe:")
    print(dynamic_universe)
