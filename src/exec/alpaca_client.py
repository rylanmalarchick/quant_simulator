import alpaca_trade_api as tradeapi
from src.config import load_config
from src.logging import get_logger
import time

logger = get_logger(__name__)

def get_alpaca_api():
    """
    Returns an Alpaca API client.
    """
    config = load_config()
    api = tradeapi.REST(
        config['api_keys']['alpaca_key'],
        config['api_keys']['alpaca_secret'],
        base_url='https://paper-api.alpaca.markets' # Use paper trading endpoint
    )
    return api

def submit_order(symbol, qty, side, type='market', time_in_force='gtc', retries=3, delay=5):
    """
    Submits an order to Alpaca.
    """
    api = get_alpaca_api()
    logger.info(f"Submitting {side} order for {qty} shares of {symbol}")
    for i in range(retries):
        try:
            order = api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force
            )
            logger.info(f"Successfully submitted order {order.id}")
            return order
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {e}")
            if i < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to submit order for {symbol} after {retries} retries.")
                return None

def get_order_status(order_id, retries=3, delay=5):
    """
    Checks the status of an order.
    """
    api = get_alpaca_api()
    logger.info(f"Getting status for order {order_id}")
    for i in range(retries):
        try:
            order = api.get_order(order_id)
            logger.info(f"Successfully fetched status for order {order_id}: {order.status}")
            return order.status
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            if i < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to get order status for {order_id} after {retries} retries.")
                return None

if __name__ == '__main__':
    # Example usage (be careful with this in a live environment)
    # order = submit_order('AAPL', 1, 'buy')
    # if order:
    #     get_order_status(order.id)
    logger.info("Alpaca client loaded.")
