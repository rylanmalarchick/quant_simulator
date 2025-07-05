import pandas as pd
from src.ingest.sqlite_writer import get_db_connection, write_features
from src.universe.merge import merge_and_tag_universe
from src.universe.rebalance import calculate_rebalance_weights
from src.logging import get_logger

logger = get_logger(__name__)

def build_and_persist_features():
    """
    Builds features for the model and persists them to the database.
    """
    logger.info("Building and persisting features...")
    conn = get_db_connection()
    
    # Load raw data
    logger.info("Loading raw data...")
    raw_bars = pd.read_sql('SELECT * FROM raw_bars', conn)
    
    # Get tagged universe and rebalance weights
    logger.info("Merging and tagging universe...")
    tagged_universe = merge_and_tag_universe()
    logger.info("Calculating rebalance weights...")
    rebalance_weights = calculate_rebalance_weights()
    
    # Feature Engineering
    logger.info("Performing feature engineering...")
    features = []
    for item in tagged_universe:
        symbol = item['symbol']
        tags = item['tags']
        
        # Filter bars for the current symbol
        symbol_bars = raw_bars[raw_bars['symbol'] == symbol].copy()
        
        if not symbol_bars.empty:
            # Multi-period returns
            symbol_bars['return_5'] = symbol_bars['close'].pct_change(5)
            symbol_bars['return_15'] = symbol_bars['close'].pct_change(15)
            symbol_bars['return_60'] = symbol_bars['close'].pct_change(60)
            
            # 20-period rolling volatility
            symbol_bars['volatility_20'] = symbol_bars['close'].rolling(20).std()
            
            # Volume/AUM ratio vs. 20-period mean
            symbol_bars['volume_ratio_20'] = symbol_bars['volume'] / symbol_bars['volume'].rolling(20).mean()
            
            # Flag features
            symbol_bars['is_pelosi'] = 1 if 'pelosi' in tags else 0
            symbol_bars['is_quantum'] = 1 if 'quantum' in tags else 0
            symbol_bars['is_dynamic'] = 1 if 'dynamic' in tags else 0
            
            # Category-weight feature
            if 'fund' in tags:
                symbol_bars['category_weight'] = rebalance_weights['funds'].get(symbol, 0)
            elif 'quantum' in tags:
                symbol_bars['category_weight'] = rebalance_weights['quantum'].get(symbol, 0)
            else:
                symbol_bars['category_weight'] = 0
            
            features.append(symbol_bars)
            
    if features:
        features_df = pd.concat(features)
        logger.info("Writing features to database...")
        write_features(features_df)
        logger.info("Successfully built and persisted features.")
        return features_df
    else:
        logger.warning("No features were built.")
        return pd.DataFrame()

if __name__ == '__main__':
    features = build_and_persist_features()
    print("Features:")
    print(features.head())
