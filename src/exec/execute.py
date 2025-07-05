from src.signal.generate import generate_signals
from src.exec.alpaca_client import submit_order
from src.exec.risk import check_risk_controls
from src.config import load_config
from src.logging import get_logger

logger = get_logger(__name__)

def execute_signals():
    """
    Executes the trading signals.
    """
    logger.info("Executing signals...")
    config = load_config()
    signals = generate_signals()
    
    # This is a placeholder for daily turnover calculation
    daily_turnover = 0.1 
    
    if signals:
        # Execute buy signals
        logger.info("Executing buy signals...")
        for _, row in signals['buy'].iterrows():
            symbol = row['symbol']
            # This is a placeholder for order size calculation
            order_size = 0.01 
            if check_risk_controls(order_size, daily_turnover):
                submit_order(symbol, 1, 'buy')
                
        # Execute sell signals
        logger.info("Executing sell signals...")
        for _, row in signals['sell'].iterrows():
            symbol = row['symbol']
            # This is a placeholder for order size calculation
            order_size = 0.01
            if check_risk_controls(order_size, daily_turnover):
                submit_order(symbol, 1, 'sell')
    else:
        logger.warning("No signals to execute.")
        
    logger.info("Finished executing signals.")

if __name__ == '__main__':
    execute_signals()
