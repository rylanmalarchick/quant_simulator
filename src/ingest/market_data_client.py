import yfinance as yf
from src.logging import get_logger
from datetime import datetime, timedelta

logger = get_logger(__name__)

def get_market_data(ticker, start_date, end_date):
    """
    Fetches historical market data from Yahoo Finance.
    """
    logger.info(f"Fetching daily data for {ticker} from {start_date} to {end_date}")
    try:
        stock = yf.Ticker(ticker)
        # Fetch daily data, which includes Open, High, Low, Close, Volume
        hist = stock.history(start=start_date, end=end_date, interval="1d")
        if hist.empty:
            logger.warning(f"No data found for {ticker} for the given date range. This could be due to an invalid ticker or no data available for the period.")
            return None
        
        # Reset index to make 'Date' a column
        hist = hist.reset_index()
        
        # Convert timestamp to unix timestamp
        hist['timestamp'] = hist['Date'].apply(lambda x: int(x.timestamp()))
        
        # Format data into the structure expected by the rest of the pipeline
        formatted_data = {
            't': hist['timestamp'].tolist(),
            'o': hist['Open'].tolist(),
            'h': hist['High'].tolist(),
            'l': hist['Low'].tolist(),
            'c': hist['Close'].tolist(),
            'v': hist['Volume'].tolist(),
            's': 'ok'
        }
        logger.info(f"Successfully fetched {len(formatted_data.get('c', []))} data points for {ticker}")
        return formatted_data
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching data for {ticker} from yfinance: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example usage
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Replace 'AAPL' with a symbol you want to test
    data = get_market_data('AAPL', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if data:
        print(f"Successfully fetched data for AAPL. First 5 close prices: {data['c'][:5]}")
    else:
        print("Failed to fetch data for AAPL.")