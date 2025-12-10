"""
Crisis/Bubble Detector using Statistical Mechanics Concepts

This module implements a bubble detection system inspired by phase transitions
in statistical mechanics. Near critical points (market crashes), systems exhibit
characteristic signatures that we can measure:

1. CORRELATION LENGTH DIVERGENCE
   - In physics: spins align over larger distances near Tc
   - In markets: stocks become more correlated before crashes
   - Measure: Average pairwise correlation of returns

2. SUSCEPTIBILITY DIVERGENCE  
   - In physics: small perturbations cause large responses
   - In markets: volatility spikes, small news moves markets
   - Measure: VIX level, realized volatility vs historical

3. CRITICAL SLOWING DOWN
   - In physics: relaxation time diverges near Tc
   - In markets: prices take longer to mean-revert
   - Measure: Autocorrelation of returns at various lags

4. FAT TAILS / KURTOSIS
   - In physics: fluctuations become non-Gaussian
   - In markets: extreme moves become more likely
   - Measure: Kurtosis of return distribution

5. MARKET BREADTH (Order Parameter)
   - In physics: magnetization in Ising model
   - In markets: % of stocks above their 200-day MA
   - Measure: Breadth indicators

Historical crisis signatures we compare against:
- Dotcom bubble (1999-2000)
- Financial crisis (2007-2008)
- COVID crash (2020)
- Current AI/Tech concentration (2024-?)

Reference papers:
- Sornette, D. "Why Stock Markets Crash" (2003)
- Johansen, A. & Sornette, D. "Critical Crashes" (1999)
- Bouchaud, J.P. "Power laws in economics" (2001)
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import json
import os
from src.logging import get_logger

logger = get_logger(__name__)

# Cache file for crisis score
CRISIS_CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'crisis_cache.json')


# Historical crisis periods for comparison
HISTORICAL_CRISES = {
    'dotcom': {
        'peak': '2000-03-10',
        'trough': '2002-10-09',
        'description': 'Dotcom bubble burst'
    },
    'financial_crisis': {
        'peak': '2007-10-09',
        'trough': '2009-03-09', 
        'description': '2008 Financial Crisis'
    },
    'covid_crash': {
        'peak': '2020-02-19',
        'trough': '2020-03-23',
        'description': 'COVID-19 crash'
    }
}

# Thresholds for warning levels (calibrated from historical data)
WARNING_THRESHOLDS = {
    'correlation': {
        'elevated': 0.5,    # Average correlation > 0.5
        'critical': 0.7     # Average correlation > 0.7 (very high)
    },
    'vix_ratio': {
        'elevated': 1.3,    # VIX 30% above 20-day MA
        'critical': 1.8     # VIX 80% above 20-day MA
    },
    'autocorrelation': {
        'elevated': 0.15,   # Significant positive autocorr
        'critical': 0.25    # Strong autocorr (critical slowing)
    },
    'kurtosis': {
        'elevated': 4.0,    # Moderate fat tails (normal = 3)
        'critical': 6.0     # Severe fat tails
    },
    'breadth': {
        'elevated': 0.4,    # Less than 40% above 200 SMA
        'critical': 0.25    # Less than 25% above 200 SMA
    }
}


class CrisisDetector:
    """
    Detects bubble/crisis conditions using statistical mechanics indicators.
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize the crisis detector.
        
        Args:
            lookback_days: Number of trading days to analyze (default 1 year)
        """
        self.lookback_days = lookback_days
        self.indicators = {}
        
    def fetch_market_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Fetch historical price data for analysis.
        Uses major indices and sector ETFs for broad market view.
        """
        if symbols is None:
            # Default: broad market representation
            symbols = [
                'SPY',   # S&P 500
                'QQQ',   # Nasdaq 100
                'IWM',   # Russell 2000
                'XLK',   # Tech sector
                'XLF',   # Financials
                'XLE',   # Energy
                'XLV',   # Healthcare
                'XLI',   # Industrials
                'XLP',   # Consumer staples
                'XLY',   # Consumer discretionary
                '^VIX',  # Volatility index
            ]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(self.lookback_days * 1.5))  # Extra for warmup
        
        logger.info(f"Fetching market data for {len(symbols)} symbols...")
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[symbol] = hist['Close']
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        
        df = pd.DataFrame(data)
        logger.info(f"Fetched {len(df)} days of data for {len(df.columns)} symbols")
        return df
    
    def calculate_correlation_indicator(self, prices: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate average pairwise correlation (correlation length proxy).
        
        In stat mech, correlation length diverges near critical point.
        High average correlation = stocks moving together = fragile market.
        """
        # Exclude VIX from correlation calculation
        price_cols = [c for c in prices.columns if c != '^VIX']
        returns = prices[price_cols].pct_change().dropna()
        
        def avg_correlation(window_returns):
            corr_matrix = window_returns.corr()
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_tri = corr_matrix.where(mask)
            return upper_tri.stack().mean()
        
        # Rolling average correlation
        avg_corr = returns.rolling(window=window).apply(
            lambda x: avg_correlation(returns.loc[x.index]), 
            raw=False
        ).iloc[:, 0]  # All columns have same correlation, take first
        
        # Simpler approach: rolling correlation matrix
        correlations = []
        for i in range(window, len(returns)):
            window_data = returns.iloc[i-window:i]
            corr_matrix = window_data.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            avg = corr_matrix.where(mask).stack().mean()
            correlations.append(avg)
        
        result = pd.Series(correlations, index=returns.index[window:])
        return result
    
    def calculate_vix_indicator(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate VIX ratio (susceptibility proxy).
        
        VIX / 20-day MA of VIX
        > 1 means elevated fear, > 1.5 means high fear
        """
        if '^VIX' not in prices.columns:
            logger.warning("VIX data not available")
            return pd.Series()
        
        vix = prices['^VIX'].dropna()
        vix_ma = vix.rolling(window=20).mean()
        vix_ratio = vix / vix_ma
        return vix_ratio
    
    def calculate_autocorrelation_indicator(self, prices: pd.DataFrame, lag: int = 1, window: int = 60) -> pd.Series:
        """
        Calculate rolling autocorrelation of returns (critical slowing down proxy).
        
        Near critical points, systems exhibit "critical slowing down" - 
        perturbations take longer to decay. In markets, this shows up as
        increased autocorrelation.
        """
        spy_returns = prices['SPY'].pct_change().dropna()
        
        def rolling_autocorr(x):
            if len(x) < lag + 2:
                return np.nan
            return x.autocorr(lag=lag)
        
        autocorr = spy_returns.rolling(window=window).apply(rolling_autocorr, raw=False)
        return autocorr
    
    def calculate_kurtosis_indicator(self, prices: pd.DataFrame, window: int = 60) -> pd.Series:
        """
        Calculate rolling kurtosis of returns (fat tails proxy).
        
        Normal distribution has kurtosis = 3.
        Higher kurtosis = fatter tails = more extreme events likely.
        Near crises, kurtosis typically increases.
        """
        spy_returns = prices['SPY'].pct_change().dropna()
        
        rolling_kurt = spy_returns.rolling(window=window).apply(
            lambda x: stats.kurtosis(x, fisher=False),  # fisher=False gives excess kurtosis + 3
            raw=True
        )
        return rolling_kurt
    
    def calculate_breadth_indicator(self, prices: pd.DataFrame, window: int = 200) -> pd.Series:
        """
        Calculate market breadth (order parameter proxy).
        
        Percentage of stocks trading above their 200-day moving average.
        Like magnetization in Ising model - measures "alignment" of stocks.
        Low breadth with high index = narrow rally = fragile.
        """
        # Exclude VIX and drop rows where all price data is NaN
        price_cols = [c for c in prices.columns if c != '^VIX']
        prices_clean = prices[price_cols].dropna(how='all')
        
        breadth_series = []
        dates = []
        for i in range(window, len(prices_clean)):
            count_above = 0
            total = 0
            for col in price_cols:
                current_price = prices_clean[col].iloc[i]
                if pd.notna(current_price):
                    sma_200 = prices_clean[col].iloc[max(0, i-window):i].mean()
                    if pd.notna(sma_200) and current_price > sma_200:
                        count_above += 1
                    total += 1
            if total > 0:
                breadth_series.append(count_above / total)
                dates.append(prices_clean.index[i])
        
        if not breadth_series:
            return pd.Series()
        
        result = pd.Series(breadth_series, index=dates)
        return result
    
    def calculate_all_indicators(self, prices: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        Calculate all crisis indicators.
        """
        if prices is None:
            prices = self.fetch_market_data()
        
        logger.info("Calculating crisis indicators...")
        
        self.indicators = {
            'correlation': self.calculate_correlation_indicator(prices),
            'vix_ratio': self.calculate_vix_indicator(prices),
            'autocorrelation': self.calculate_autocorrelation_indicator(prices),
            'kurtosis': self.calculate_kurtosis_indicator(prices),
            'breadth': self.calculate_breadth_indicator(prices),
        }
        
        return self.indicators
    
    def get_current_readings(self) -> Dict[str, float]:
        """
        Get the most recent reading for each indicator.
        """
        if not self.indicators:
            self.calculate_all_indicators()
        
        readings = {}
        for name, series in self.indicators.items():
            if len(series) > 0:
                readings[name] = series.iloc[-1]
            else:
                readings[name] = np.nan
        
        return readings
    
    def assess_warning_level(self, readings: Dict[str, float] = None) -> Dict[str, str]:
        """
        Assess warning level for each indicator.
        
        Returns: Dict with 'normal', 'elevated', or 'critical' for each indicator
        """
        if readings is None:
            readings = self.get_current_readings()
        
        levels = {}
        
        for indicator, value in readings.items():
            if pd.isna(value):
                levels[indicator] = 'unknown'
                continue
                
            thresholds = WARNING_THRESHOLDS.get(indicator, {})
            
            if indicator == 'breadth':
                # Breadth is inverted - low is bad
                if value < thresholds.get('critical', 0.25):
                    levels[indicator] = 'critical'
                elif value < thresholds.get('elevated', 0.4):
                    levels[indicator] = 'elevated'
                else:
                    levels[indicator] = 'normal'
            else:
                # Other indicators - high is bad
                if value > thresholds.get('critical', float('inf')):
                    levels[indicator] = 'critical'
                elif value > thresholds.get('elevated', float('inf')):
                    levels[indicator] = 'elevated'
                else:
                    levels[indicator] = 'normal'
        
        return levels
    
    def calculate_composite_score(self, readings: Dict[str, float] = None) -> float:
        """
        Calculate composite crisis score (0-100).
        
        0 = calm markets, 100 = extreme crisis conditions
        """
        if readings is None:
            readings = self.get_current_readings()
        
        scores = []
        weights = {
            'correlation': 0.25,
            'vix_ratio': 0.25,
            'autocorrelation': 0.15,
            'kurtosis': 0.15,
            'breadth': 0.20,
        }
        
        for indicator, value in readings.items():
            if pd.isna(value) or indicator not in weights:
                continue
            
            thresholds = WARNING_THRESHOLDS.get(indicator, {})
            elevated = thresholds.get('elevated', 1)
            critical = thresholds.get('critical', 2)
            
            if indicator == 'breadth':
                # Invert breadth (lower is worse)
                # Map 0.25-0.6 to 0-100
                normalized = max(0, min(100, (0.6 - value) / 0.35 * 100))
            else:
                # Map from 0 to critical threshold
                normalized = max(0, min(100, value / critical * 100))
            
            scores.append(normalized * weights[indicator])
        
        if not scores:
            return 0
        
        return sum(scores) / sum(weights[k] for k in readings.keys() if k in weights)
    
    def generate_report(self) -> str:
        """
        Generate a human-readable crisis assessment report.
        """
        readings = self.get_current_readings()
        levels = self.assess_warning_level(readings)
        composite = self.calculate_composite_score(readings)
        
        report = []
        report.append("=" * 60)
        report.append("CRISIS/BUBBLE DETECTOR - STATISTICAL MECHANICS ANALYSIS")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)
        report.append("")
        
        # Composite score
        if composite < 30:
            status = "CALM"
            emoji = ""
        elif composite < 50:
            status = "ELEVATED"
            emoji = ""
        elif composite < 70:
            status = "HIGH ALERT"
            emoji = ""
        else:
            status = "CRITICAL"
            emoji = ""
        
        report.append(f"COMPOSITE CRISIS SCORE: {composite:.1f}/100 - {status}")
        report.append("")
        report.append("INDIVIDUAL INDICATORS:")
        report.append("-" * 40)
        
        indicator_descriptions = {
            'correlation': 'Cross-Asset Correlation (herd behavior)',
            'vix_ratio': 'VIX Ratio (fear level)',
            'autocorrelation': 'Return Autocorrelation (critical slowing)',
            'kurtosis': 'Return Kurtosis (fat tails)',
            'breadth': 'Market Breadth (participation)',
        }
        
        for indicator, value in readings.items():
            level = levels.get(indicator, 'unknown')
            desc = indicator_descriptions.get(indicator, indicator)
            
            if level == 'critical':
                level_str = "[CRITICAL]"
            elif level == 'elevated':
                level_str = "[ELEVATED]"
            else:
                level_str = "[NORMAL]"
            
            if pd.notna(value):
                report.append(f"  {desc}")
                report.append(f"    Value: {value:.3f} {level_str}")
            else:
                report.append(f"  {desc}: N/A")
        
        report.append("")
        report.append("INTERPRETATION:")
        report.append("-" * 40)
        
        if composite >= 70:
            report.append("  Market is showing multiple crisis signatures.")
            report.append("  Consider reducing risk exposure significantly.")
            report.append("  Historical parallels: Late 2007, Feb 2020")
        elif composite >= 50:
            report.append("  Elevated stress indicators detected.")
            report.append("  Consider defensive positioning.")
            report.append("  Monitor closely for deterioration.")
        elif composite >= 30:
            report.append("  Some stress indicators elevated.")
            report.append("  Normal market fluctuations, but stay alert.")
        else:
            report.append("  Markets appear relatively calm.")
            report.append("  No immediate crisis signatures detected.")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def run_crisis_check() -> Dict:
    """
    Run a full crisis check and return results.
    """
    detector = CrisisDetector()
    detector.calculate_all_indicators()
    
    readings = detector.get_current_readings()
    levels = detector.assess_warning_level(readings)
    composite = detector.calculate_composite_score(readings)
    report = detector.generate_report()
    
    return {
        'readings': readings,
        'levels': levels,
        'composite_score': composite,
        'report': report
    }


def save_crisis_cache(result: Dict) -> str:
    """
    Save crisis check results to cache file.
    Converts numpy types to Python types for JSON serialization.
    """
    # Convert numpy types to Python types
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'composite_score': float(result['composite_score']),
        'levels': result['levels'],
        'readings': {k: float(v) if pd.notna(v) else None for k, v in result['readings'].items()},
    }
    
    # Determine overall status from composite score
    score = cache_data['composite_score']
    if score < 30:
        cache_data['status'] = 'NORMAL'
    elif score < 50:
        cache_data['status'] = 'ELEVATED'
    elif score < 70:
        cache_data['status'] = 'HIGH'
    else:
        cache_data['status'] = 'CRITICAL'
    
    with open(CRISIS_CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    logger.info(f"Crisis cache saved: score={score:.1f}, status={cache_data['status']}")
    return CRISIS_CACHE_FILE


def load_crisis_cache() -> Optional[Dict]:
    """
    Load cached crisis data. Returns None if cache is missing or stale (not from today).
    """
    try:
        if not os.path.exists(CRISIS_CACHE_FILE):
            logger.info("No crisis cache file found")
            return None
        
        with open(CRISIS_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is from today
        cache_date = cache_data.get('date', '')
        today = datetime.now().strftime('%Y-%m-%d')
        
        if cache_date != today:
            logger.info(f"Crisis cache is stale (from {cache_date})")
            return None
        
        logger.info(f"Loaded crisis cache: score={cache_data.get('composite_score', 0):.1f}")
        return cache_data
        
    except Exception as e:
        logger.error(f"Failed to load crisis cache: {e}")
        return None


def get_crisis_score(force_refresh: bool = False) -> Dict:
    """
    Get the crisis score, using cache if available and fresh.
    
    Args:
        force_refresh: If True, always run fresh calculation
        
    Returns:
        Dict with composite_score, status, levels, readings, timestamp
    """
    if not force_refresh:
        cached = load_crisis_cache()
        if cached:
            return cached
    
    # Run fresh calculation
    logger.info("Running fresh crisis check...")
    result = run_crisis_check()
    save_crisis_cache(result)
    
    return load_crisis_cache()


if __name__ == '__main__':
    result = run_crisis_check()
    print(result['report'])
    save_crisis_cache(result)
