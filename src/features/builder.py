import pandas as pd
import numpy as np
from src.ingest.sqlite_writer import get_db_connection, write_features
from src.universe.merge import merge_and_tag_universe
from src.universe.rebalance import calculate_rebalance_weights
from src.logging import get_logger

logger = get_logger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss over the period
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    # Use exponential moving average for smoothing
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_pct_b(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Calculate Bollinger %B.
    %B = (Price - Lower Band) / (Upper Band - Lower Band)
    Values: 0 = at lower band, 1 = at upper band, >1 = above upper, <0 = below lower
    """
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    
    pct_b = (prices - lower_band) / (upper_band - lower_band)
    return pct_b


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = EMA of True Range
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr


def calculate_sma_cross(prices: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """
    Calculate SMA Cross signal.
    Returns: ratio of fast SMA to slow SMA (>1 = bullish, <1 = bearish)
    """
    sma_fast = prices.rolling(window=fast).mean()
    sma_slow = prices.rolling(window=slow).mean()
    
    # Return ratio instead of binary to give model more info
    sma_cross_ratio = sma_fast / sma_slow
    return sma_cross_ratio


# === TIER 2 TECHNICAL INDICATORS ===

def calculate_obv_slope(close: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV) slope.
    OBV accumulates volume based on price direction.
    Slope = linear regression slope of OBV over period (normalized).
    """
    # Calculate OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    # Calculate slope using rolling linear regression
    def rolling_slope(x):
        if len(x) < period:
            return np.nan
        n = len(x)
        x_vals = np.arange(n)
        slope = np.polyfit(x_vals, x, 1)[0]
        return slope
    
    obv_slope = obv.rolling(window=period).apply(rolling_slope, raw=True)
    
    # Normalize by average volume to make it comparable across stocks
    avg_volume = volume.rolling(window=period).mean()
    obv_slope_normalized = obv_slope / avg_volume.replace(0, np.nan)
    
    return obv_slope_normalized


def calculate_price_vs_sma200(prices: pd.Series) -> pd.Series:
    """
    Calculate price distance from 200-day SMA.
    Returns: (price - SMA200) / SMA200 as percentage.
    Positive = above long-term trend, negative = below.
    """
    sma_200 = prices.rolling(window=200).mean()
    price_vs_sma200 = (prices - sma_200) / sma_200
    return price_vs_sma200


def calculate_52week_high_ratio(prices: pd.Series) -> pd.Series:
    """
    Calculate ratio of current price to 52-week high.
    Returns: current price / 252-day rolling max (trading days in a year).
    1.0 = at 52-week high, lower = further from high.
    """
    rolling_max_252 = prices.rolling(window=252, min_periods=50).max()
    high_ratio = prices / rolling_max_252
    return high_ratio


def calculate_overnight_gap(open_price: pd.Series, prev_close: pd.Series) -> pd.Series:
    """
    Calculate overnight gap (open vs previous close).
    Returns: (open - prev_close) / prev_close as percentage.
    Positive = gap up, negative = gap down.
    """
    gap = (open_price - prev_close) / prev_close
    return gap


def calculate_day_of_week(timestamps: pd.Series) -> tuple:
    """
    Calculate day of week with cyclical encoding.
    Returns: (sin_dow, cos_dow) for cyclical representation.
    This preserves the cyclical nature (Friday is close to Monday).
    """
    # Convert to datetime if needed
    if timestamps.dtype != 'datetime64[ns]':
        dt = pd.to_datetime(timestamps, unit='s')
    else:
        dt = timestamps
    
    # Get day of week (0=Monday, 6=Sunday)
    dow = dt.dt.dayofweek
    
    # Cyclical encoding (5 trading days: 0-4)
    sin_dow = np.sin(2 * np.pi * dow / 5)
    cos_dow = np.cos(2 * np.pi * dow / 5)
    
    return sin_dow, cos_dow


# === TIER 3 MARKET CONTEXT INDICATORS ===

# Sector mapping for sector momentum calculation
SECTOR_MAPPING = {
    # Tech / Semiconductors
    'NVDA': 'semiconductors',
    'AMD': 'semiconductors', 
    'MRVL': 'semiconductors',
    'QRVO': 'semiconductors',
    'LSCC': 'semiconductors',
    'KLIC': 'semiconductors',
    # Big Tech
    'MSFT': 'big_tech',
    'GOOG': 'big_tech',
    'AMZN': 'big_tech',
    # Quantum-focused
    'IONQ': 'quantum',
    'QBTS': 'quantum',
    'IBM': 'quantum',  # IBM has quantum division
    # Industrials
    'HON': 'industrials',
}

def build_and_persist_features():
    """
    Builds features for the model and persists them to the database.
    """
    logger.info("Building and persisting features...")
    conn = get_db_connection()
    
    # Load raw data
    logger.info("Loading raw data...")
    raw_bars = pd.read_sql('SELECT * FROM raw_bars', conn)
    
    # === TIER 3: Load market benchmark data ===
    logger.info("Loading market benchmark data (SPY, VIX)...")
    
    # Load SPY for market regime and correlation features
    spy_bars = raw_bars[raw_bars['symbol'] == 'SPY'].copy()
    if not spy_bars.empty:
        spy_bars = spy_bars.sort_values('timestamp').reset_index(drop=True)
        spy_bars['spy_return'] = spy_bars['close'].pct_change()
        spy_bars['spy_sma_200'] = spy_bars['close'].rolling(window=200).mean()
        spy_bars['market_regime'] = (spy_bars['close'] > spy_bars['spy_sma_200']).astype(int)  # 1 = bull, 0 = bear
        spy_data = spy_bars[['timestamp', 'close', 'spy_return', 'market_regime']].rename(
            columns={'close': 'spy_close'}
        )
        logger.info(f"Loaded {len(spy_bars)} SPY bars for market context")
    else:
        spy_data = None
        logger.warning("No SPY data found - market regime features will be skipped")
    
    # Load VIX for volatility regime features
    vix_bars = raw_bars[raw_bars['symbol'] == '^VIX'].copy()
    if not vix_bars.empty:
        vix_bars = vix_bars.sort_values('timestamp').reset_index(drop=True)
        vix_bars['vix_sma_20'] = vix_bars['close'].rolling(window=20).mean()
        vix_bars['vix_ratio'] = vix_bars['close'] / vix_bars['vix_sma_20']  # >1 = high fear, <1 = complacent
        vix_data = vix_bars[['timestamp', 'close', 'vix_ratio']].rename(
            columns={'close': 'vix_close'}
        )
        logger.info(f"Loaded {len(vix_bars)} VIX bars for volatility context")
    else:
        vix_data = None
        logger.warning("No VIX data found - VIX ratio features will be skipped")
    
    # Pre-calculate sector returns for sector momentum feature
    logger.info("Pre-calculating sector returns...")
    sector_returns = {}
    for symbol, sector in SECTOR_MAPPING.items():
        sector_bars = raw_bars[raw_bars['symbol'] == symbol].copy()
        if not sector_bars.empty:
            sector_bars = sector_bars.sort_values('timestamp').reset_index(drop=True)
            sector_bars['return_5d'] = sector_bars['close'].pct_change(5)
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(
                sector_bars[['timestamp', 'return_5d']].rename(columns={'return_5d': f'{symbol}_ret'})
            )
    
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
            # Sort by timestamp to ensure correct order
            symbol_bars = symbol_bars.sort_values('timestamp').reset_index(drop=True)
            
            # Multi-period returns
            symbol_bars['return_5'] = symbol_bars['close'].pct_change(5)
            symbol_bars['return_15'] = symbol_bars['close'].pct_change(15)
            symbol_bars['return_60'] = symbol_bars['close'].pct_change(60)
            
            # 20-period rolling volatility
            symbol_bars['volatility_20'] = symbol_bars['close'].rolling(20).std()
            
            # Volume/AUM ratio vs. 20-period mean
            symbol_bars['volume_ratio_20'] = symbol_bars['volume'] / symbol_bars['volume'].rolling(20).mean()
            
            # === TIER 1 TECHNICAL INDICATORS ===
            
            # RSI (14) - Relative Strength Index
            symbol_bars['rsi_14'] = calculate_rsi(symbol_bars['close'], period=14)
            
            # MACD - Moving Average Convergence Divergence
            macd_line, macd_signal, macd_hist = calculate_macd(symbol_bars['close'])
            symbol_bars['macd_line'] = macd_line
            symbol_bars['macd_signal'] = macd_signal
            symbol_bars['macd_histogram'] = macd_hist
            
            # Bollinger %B
            symbol_bars['bollinger_pct_b'] = calculate_bollinger_pct_b(symbol_bars['close'])
            
            # ATR (14) - Average True Range (normalized by close price for comparability)
            atr = calculate_atr(symbol_bars['high'], symbol_bars['low'], symbol_bars['close'])
            symbol_bars['atr_14'] = atr
            symbol_bars['atr_pct'] = atr / symbol_bars['close']  # ATR as % of price
            
            # SMA Cross (10/50)
            symbol_bars['sma_cross_ratio'] = calculate_sma_cross(symbol_bars['close'])
            
            # === END TIER 1 ===
            
            # === TIER 2 TECHNICAL INDICATORS ===
            
            # OBV Slope - Volume confirms price moves
            symbol_bars['obv_slope'] = calculate_obv_slope(symbol_bars['close'], symbol_bars['volume'])
            
            # Price vs SMA200 - Long-term trend context
            symbol_bars['price_vs_sma200'] = calculate_price_vs_sma200(symbol_bars['close'])
            
            # 52-Week High Ratio - Momentum/breakout potential
            symbol_bars['week52_high_ratio'] = calculate_52week_high_ratio(symbol_bars['close'])
            
            # Overnight Gap - Overnight sentiment
            symbol_bars['overnight_gap'] = calculate_overnight_gap(
                symbol_bars['open'], 
                symbol_bars['close'].shift(1)
            )
            
            # Day of Week - Cyclical encoding for calendar anomalies
            sin_dow, cos_dow = calculate_day_of_week(symbol_bars['timestamp'])
            symbol_bars['day_of_week_sin'] = sin_dow
            symbol_bars['day_of_week_cos'] = cos_dow
            
            # === END TIER 2 ===
            
            # === TIER 3: MARKET CONTEXT INDICATORS ===
            
            # Market Regime (SPY > SMA200) - Bull/Bear market context
            if spy_data is not None:
                symbol_bars = symbol_bars.merge(
                    spy_data[['timestamp', 'spy_return', 'market_regime']], 
                    on='timestamp', 
                    how='left'
                )
                # Correlation to SPY (rolling 20-day)
                symbol_bars['spy_return'] = symbol_bars['spy_return'].fillna(0)
                stock_returns = symbol_bars['close'].pct_change()
                symbol_bars['corr_to_spy'] = stock_returns.rolling(window=20).corr(symbol_bars['spy_return'])
            else:
                symbol_bars['market_regime'] = np.nan
                symbol_bars['spy_return'] = np.nan
                symbol_bars['corr_to_spy'] = np.nan
            
            # VIX Ratio - Volatility regime (fear vs complacency)
            if vix_data is not None:
                symbol_bars = symbol_bars.merge(
                    vix_data[['timestamp', 'vix_ratio']], 
                    on='timestamp', 
                    how='left'
                )
            else:
                symbol_bars['vix_ratio'] = np.nan
            
            # Sector Momentum - Average 5-day return of sector peers (excluding self)
            current_sector = SECTOR_MAPPING.get(symbol)
            if current_sector and current_sector in sector_returns:
                # Get returns from other stocks in the same sector
                peer_returns = []
                for peer_df in sector_returns[current_sector]:
                    # Check if this is not the current symbol
                    col_name = [c for c in peer_df.columns if c.endswith('_ret')][0]
                    peer_symbol = col_name.replace('_ret', '')
                    if peer_symbol != symbol:
                        peer_returns.append(peer_df.rename(columns={col_name: 'peer_ret'}))
                
                if peer_returns:
                    # Merge all peer returns and calculate average
                    merged_peers = symbol_bars[['timestamp']].copy()
                    for i, pr in enumerate(peer_returns):
                        merged_peers = merged_peers.merge(
                            pr.rename(columns={'peer_ret': f'peer_ret_{i}'}),
                            on='timestamp',
                            how='left'
                        )
                    # Calculate average sector momentum (excluding self)
                    peer_cols = [c for c in merged_peers.columns if c.startswith('peer_ret_')]
                    if peer_cols:
                        symbol_bars['sector_momentum'] = merged_peers[peer_cols].mean(axis=1)
                    else:
                        symbol_bars['sector_momentum'] = np.nan
                else:
                    symbol_bars['sector_momentum'] = np.nan
            else:
                symbol_bars['sector_momentum'] = np.nan
            
            # === END TIER 3 ===
            
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
