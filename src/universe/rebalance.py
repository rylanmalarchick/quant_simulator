import pandas as pd
from src.ingest import market_data_client # Use the new client
from src.universe.loader import load_and_validate_universe
from datetime import datetime, timedelta
from src.logging import get_logger

logger = get_logger(__name__)

def calculate_rebalance_weights():
    """
    Calculates the rebalance weights for the static categories (funds and quantum).
    """
    logger.info("Calculating rebalance weights...")
    universe = load_and_validate_universe()
    
    # Calculate 30-day returns for funds
    fund_returns = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for ticker in universe.get('funds', []):
        try:
            candles = market_data_client.get_market_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if candles and candles.get('s') == 'ok' and len(candles['c']) > 1:
                price_30_days_ago = candles['c'][0]
                current_price = candles['c'][-1]
                fund_returns.append((ticker, (current_price - price_30_days_ago) / price_30_days_ago))
        except Exception as e:
            logger.error(f"Could not get data for {ticker} for rebalance calculation: {e}")

    # Calculate 30-day returns for quantum
    quantum_returns = []
    for ticker in universe.get('quantum', []):
        try:
            candles = market_data_client.get_market_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if candles and candles.get('s') == 'ok' and len(candles['c']) > 1:
                price_30_days_ago = candles['c'][0]
                current_price = candles['c'][-1]
                quantum_returns.append((ticker, (current_price - price_30_days_ago) / price_30_days_ago))
        except Exception as e:
            logger.error(f"Could not get data for {ticker} for rebalance calculation: {e}")
            
    # Calculate relative weights
    fund_total_return = sum(r for _, r in fund_returns)
    quantum_total_return = sum(r for _, r in quantum_returns)

    fund_weights = {ticker: ret / fund_total_return if fund_total_return else 0 for ticker, ret in fund_returns}
    quantum_weights = {ticker: ret / quantum_total_return if quantum_total_return else 0 for ticker, ret in quantum_returns}
    
    logger.info("Successfully calculated rebalance weights.")
    return {'funds': fund_weights, 'quantum': quantum_weights}

if __name__ == '__main__':
    rebalance_weights = calculate_rebalance_weights()
    print("Rebalance weights:")
    print(rebalance_weights)
