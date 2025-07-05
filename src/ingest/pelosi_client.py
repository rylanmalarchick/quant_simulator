import requests
import json
from src.logging import get_logger
import time

logger = get_logger(__name__)

def get_pelosi_trades(retries=3, delay=5):
    """
    Fetches and filters Nancy Pelosi's trades from the House Stock Watcher JSON feed.
    """
    url = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/stock_trades/all_transactions.json"
    logger.info("Fetching Pelosi trades")
    for i in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            all_transactions = response.json()
            
            pelosi_trades = [
                t for t in all_transactions 
                if t['representative'].strip() == 'Nancy Pelosi'
            ]
            
            logger.info(f"Successfully fetched {len(pelosi_trades)} Pelosi trades.")
            return pelosi_trades
        except Exception as e:
            logger.error(f"Error fetching Pelosi trades: {e}")
            if i < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to fetch Pelosi trades after {retries} retries.")
                return []

if __name__ == '__main__':
    # Example usage
    trades = get_pelosi_trades()
    print(json.dumps(trades, indent=2))