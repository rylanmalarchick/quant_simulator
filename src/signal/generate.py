import pandas as pd
import pickle
import os
from datetime import datetime
from src.features.builder import build_and_persist_features
from src.config import load_config
from src.ingest.sqlite_writer import get_db_connection
from src.logging import get_logger

logger = get_logger(__name__)

def generate_signals():
    """
    Generates trading signals.
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
    
    # Rank and select signals
    logger.info("Ranking and selecting signals...")
    latest_features['buy_prob'] = probabilities
    top_k = config['top_k']
    
    buy_signals = latest_features.nlargest(top_k, 'buy_prob')
    sell_signals = latest_features.nsmallest(top_k, 'buy_prob')
    
    # Log signals
    log_message = f"--- Signals for {datetime.now()} ---\n"
    log_message += "BUY:\n"
    log_message += buy_signals[['symbol', 'buy_prob']].to_string()
    log_message += "\nSELL:\n"
    log_message += sell_signals[['symbol', 'buy_prob']].to_string()
    
    logger.info(log_message)
    
    with open('signals.log', 'a') as f:
        f.write(log_message + "\n\n")
        
    logger.info("Successfully generated signals.")
    return {'buy': buy_signals, 'sell': sell_signals}

if __name__ == '__main__':
    generate_signals()
