import pandas as pd
import pickle
import os
from datetime import datetime
from src.features.builder import build_and_persist_features
from src.config import load_config
from src.ingest.sqlite_writer import get_db_connection
from src.logging import get_logger

logger = get_logger(__name__)

# Confidence thresholds for filtering low-conviction trades
# Only trade when model is confident (reduces noise, improves win rate)
CONFIDENCE_THRESHOLD_BUY = 0.60   # Must be >60% confident to buy
CONFIDENCE_THRESHOLD_SELL = 0.40  # Must be <40% confident to short (i.e., 60%+ confident it will fall)


def generate_signals():
    """
    Generates trading signals with confidence filtering.
    
    CONFIDENCE FILTERING:
    - Only generates BUY signals when probability > CONFIDENCE_THRESHOLD_BUY
    - Only generates SELL signals when probability < CONFIDENCE_THRESHOLD_SELL
    - Stocks in the "uncertain zone" (40-60%) are skipped entirely
    - This reduces trading frequency but improves signal quality
    """
    logger.info("Generating signals...")
    config = load_config()
    
    # Re-ingest latest data and build features
    logger.info("Re-ingesting latest data and building features...")
    build_and_persist_features()
    
    # Load latest model
    logger.info("Loading latest model...")
    model_dir = 'models'
    try:
        latest_model_path = max([os.path.join(model_dir, f) for f in os.listdir(model_dir)], key=os.path.getctime)
        with open(latest_model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded model: {latest_model_path}")
    except (FileNotFoundError, ValueError):
        logger.error("No model found. Please train a model first.")
        return None
        
    # Load features
    logger.info("Loading features...")
    conn = get_db_connection()
    features_df = pd.read_sql('SELECT * FROM features', conn)
    
    # Get latest features for each symbol
    logger.info("Getting latest features for each symbol...")
    latest_features = features_df.groupby('symbol').last().reset_index()
    
    # Predict probabilities
    logger.info("Predicting probabilities...")
    X = latest_features.drop(['symbol', 'timestamp', 'close'], axis=1)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Rank and select signals with confidence filtering
    logger.info("Ranking and selecting signals (with confidence filtering)...")
    latest_features['buy_prob'] = probabilities
    top_k = config['top_k']
    
    # Apply confidence filtering
    high_confidence_buys = latest_features[latest_features['buy_prob'] > CONFIDENCE_THRESHOLD_BUY]
    high_confidence_sells = latest_features[latest_features['buy_prob'] < CONFIDENCE_THRESHOLD_SELL]
    
    # Select top_k from filtered results
    buy_signals = high_confidence_buys.nlargest(top_k, 'buy_prob')
    sell_signals = high_confidence_sells.nsmallest(top_k, 'buy_prob')
    
    # Log confidence stats
    total_symbols = len(latest_features)
    uncertain_count = total_symbols - len(high_confidence_buys) - len(high_confidence_sells)
    logger.info(f"Confidence filtering: {len(high_confidence_buys)} buys, {len(high_confidence_sells)} sells, {uncertain_count} uncertain (skipped)")
    
    # Log signals
    log_message = f"--- Signals for {datetime.now()} ---\n"
    log_message += f"Confidence thresholds: BUY>{CONFIDENCE_THRESHOLD_BUY}, SELL<{CONFIDENCE_THRESHOLD_SELL}\n"
    log_message += f"Filtered: {len(high_confidence_buys)} potential buys, {len(high_confidence_sells)} potential sells\n"
    log_message += "BUY:\n"
    if len(buy_signals) > 0:
        log_message += buy_signals[['symbol', 'buy_prob']].to_string()
    else:
        log_message += "  (no high-confidence buy signals)"
    log_message += "\nSELL:\n"
    if len(sell_signals) > 0:
        log_message += sell_signals[['symbol', 'buy_prob']].to_string()
    else:
        log_message += "  (no high-confidence sell signals)"
    
    logger.info(log_message)
    
    with open('signals.log', 'a') as f:
        f.write(log_message + "\n\n")
        
    logger.info("Successfully generated signals.")
    return {'buy': buy_signals, 'sell': sell_signals}

if __name__ == '__main__':
    generate_signals()
