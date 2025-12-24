"""
Walk-Forward Backtesting Engine v2 - Enhanced for quantSim

IMPROVEMENTS OVER V1:
1. TARGET: 5-day forward risk-adjusted returns (not binary next-day)
2. FEATURES: 50+ sophisticated indicators with Numba acceleration
3. REGIME: Market regime detection (bull/bear/sideways) with adaptive behavior
4. SIZING: Conviction-weighted position sizing
5. HARDWARE: GPU via XGBoost CUDA, 32-core parallel processing

Hardware targets:
- CPU: Intel i9-13980HX (32 threads)
- GPU: NVIDIA RTX 4070 Laptop (8GB VRAM, CUDA 13.0)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from numba import jit, prange
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

from src.logging import get_logger

logger = get_logger(__name__)

# Number of CPU cores to use
N_JOBS = 28  # Leave 4 cores for system on 32-thread CPU


# ============================================================================
# NUMBA-ACCELERATED TECHNICAL INDICATORS
# ============================================================================

@jit(nopython=True, cache=True)
def calc_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average - Numba accelerated."""
    result = np.empty_like(prices)
    result[:period] = np.nan
    multiplier = 2.0 / (period + 1)
    result[period-1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        result[i] = (prices[i] - result[i-1]) * multiplier + result[i-1]
    return result


@jit(nopython=True, cache=True)
def calc_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI - Numba accelerated."""
    n = len(prices)
    result = np.empty(n)
    result[:] = np.nan
    
    if n < period + 1:
        return result
    
    deltas = np.empty(n)
    deltas[0] = 0
    for i in range(1, n):
        deltas[i] = prices[i] - prices[i-1]
    
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


@jit(nopython=True, cache=True)
def calc_macd(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD with signal and histogram - Numba accelerated."""
    ema12 = calc_ema(prices, 12)
    ema26 = calc_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


@jit(nopython=True, cache=True)
def calc_bollinger(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands - Numba accelerated."""
    n = len(prices)
    middle = np.empty(n)
    upper = np.empty(n)
    lower = np.empty(n)
    middle[:] = np.nan
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_dev * std
        lower[i] = middle[i] - std_dev * std
    
    return upper, middle, lower


@jit(nopython=True, cache=True)
def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range - Numba accelerated."""
    n = len(close)
    tr = np.empty(n)
    atr = np.empty(n)
    tr[:] = np.nan
    atr[:] = np.nan
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


@jit(nopython=True, cache=True)
def calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index - Numba accelerated."""
    n = len(close)
    adx = np.empty(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0.0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0.0:
            minus_dm[i] = down_move
    
    atr = calc_atr(high, low, close, period)
    
    # Smooth DM
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period + 1, n):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr[i] / period
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr[i] / period
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period*2])
    for i in range(period * 2, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx


@jit(nopython=True, cache=True)  
def calc_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume - Numba accelerated."""
    n = len(close)
    obv = np.zeros(n)
    obv[0] = volume[0]
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    return obv


@jit(nopython=True, cache=True)
def calc_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Volume Weighted Average Price - Numba accelerated."""
    typical_price = (high + low + close) / 3.0
    cumulative_tp_vol = np.cumsum(typical_price * volume)
    cumulative_vol = np.cumsum(volume)
    
    vwap = np.empty_like(close)
    for i in range(len(close)):
        if cumulative_vol[i] > 0:
            vwap[i] = cumulative_tp_vol[i] / cumulative_vol[i]
        else:
            vwap[i] = close[i]
    
    return vwap


@jit(nopython=True, cache=True)
def calc_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator - Numba accelerated."""
    n = len(close)
    k = np.empty(n)
    k[:] = np.nan
    
    for i in range(k_period - 1, n):
        highest_high = np.max(high[i - k_period + 1:i + 1])
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        
        if highest_high != lowest_low:
            k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            k[i] = 50.0
    
    # %D is SMA of %K
    d = np.empty(n)
    d[:] = np.nan
    for i in range(k_period + d_period - 2, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    conviction: float = 0.5  # Model confidence
    side: str = 'long'
    high_water_mark: float = 0.0  # For trailing stop
    
    def __post_init__(self):
        if self.high_water_mark == 0.0:
            self.high_water_mark = self.entry_price
    
    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price
    
    def update_high_water_mark(self, current_price: float) -> None:
        """Update the high water mark for trailing stop."""
        if current_price > self.high_water_mark:
            self.high_water_mark = current_price
    
    def check_trailing_stop(self, current_price: float, trailing_pct: float = 0.08) -> bool:
        """Check if trailing stop is hit. Returns True if should exit."""
        if self.high_water_mark <= 0:
            return False
        drawdown_from_high = (self.high_water_mark - current_price) / self.high_water_mark
        return drawdown_from_high >= trailing_pct


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str
    shares: int
    entry_price: float
    exit_price: float
    entry_date: datetime
    exit_date: datetime
    pnl: float
    pnl_percent: float
    holding_days: int
    conviction: float
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'shares': self.shares,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'exit_date': self.exit_date.strftime('%Y-%m-%d'),
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'holding_days': self.holding_days,
            'conviction': self.conviction,
        }


@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str  # 'bull', 'bear', 'sideways'
    strength: float  # 0-1, how strong the regime signal is
    volatility: str  # 'low', 'medium', 'high'
    trend_strength: float  # ADX-based
    
    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest."""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    benchmark_return: float
    benchmark_return_pct: float
    alpha: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float  # NEW: annualized return / max drawdown
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    volatility: float
    downside_volatility: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_period: float
    expectancy: float  # NEW: expected $ per trade
    
    # Statistical tests
    t_statistic: float
    p_value: float
    
    # NEW: Regime performance
    bull_return: float
    bear_return: float
    sideways_return: float
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class WalkForwardResult:
    """Complete results from walk-forward backtest."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    
    # Time series
    equity_curve: pd.DataFrame
    benchmark_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    
    # Performance
    metrics: BacktestMetrics
    trades: List[Trade]
    
    # Walk-forward specific
    n_train_periods: int
    train_window_days: int
    test_window_days: int
    
    # NEW: Regime history
    regime_history: List[Dict]
    
    def summary(self) -> str:
        """Return formatted summary."""
        m = self.metrics
        return f"""
================================================================================
                    WALK-FORWARD BACKTEST RESULTS (V2 - ENHANCED)
================================================================================
Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
Initial Capital: ${self.initial_capital:,.2f}
Final Value: ${self.final_value:,.2f}

Walk-Forward Parameters:
  Training Window: {self.train_window_days} days
  Testing Window: {self.test_window_days} days  
  Total Periods: {self.n_train_periods}

--- RETURNS ---
Total Return:       ${m.total_return:,.2f} ({m.total_return_pct:+.2%})
Annualized Return:  {m.annualized_return:+.2%}
Benchmark (SPY):    {m.benchmark_return_pct:+.2%}
Alpha:              {m.alpha:+.2%}

--- RISK METRICS ---
Sharpe Ratio:       {m.sharpe_ratio:.2f}
Sortino Ratio:      {m.sortino_ratio:.2f}
Calmar Ratio:       {m.calmar_ratio:.2f}
Max Drawdown:       ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2%})
Max DD Duration:    {m.max_drawdown_duration} days
Volatility (ann.):  {m.volatility:.2%}

--- TRADE STATISTICS ---
Total Trades:       {m.total_trades}
Win Rate:           {m.win_rate:.2%}
Avg Win:            ${m.avg_win:,.2f}
Avg Loss:           ${m.avg_loss:,.2f}
Profit Factor:      {m.profit_factor:.2f}
Expectancy:         ${m.expectancy:,.2f} per trade
Avg Holding Period: {m.avg_holding_period:.1f} days

--- REGIME PERFORMANCE ---
Bull Markets:       {m.bull_return:+.2%}
Bear Markets:       {m.bear_return:+.2%}
Sideways Markets:   {m.sideways_return:+.2%}

--- STATISTICAL SIGNIFICANCE ---
T-Statistic vs SPY: {m.t_statistic:.2f}
P-Value:            {m.p_value:.4f}
Significant (p<0.05): {'YES ✓' if m.p_value < 0.05 else 'NO'}

================================================================================
"""

    def to_json(self) -> str:
        """Export results to JSON."""
        return json.dumps({
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'final_value': self.final_value,
            'metrics': self.metrics.to_dict(),
            'n_train_periods': self.n_train_periods,
            'train_window_days': self.train_window_days,
            'test_window_days': self.test_window_days,
            'trades': [t.to_dict() for t in self.trades],
        }, indent=2, default=str)


# ============================================================================
# ENHANCED FEATURE CALCULATOR
# ============================================================================

class FeatureCalculatorV2:
    """
    Enhanced feature calculator with 50+ indicators.
    Uses Numba for acceleration and parallel processing.
    """
    
    def __init__(self):
        self.spy_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None
        self.current_regime: Optional[MarketRegime] = None
    
    def set_market_data(self, spy_df: pd.DataFrame, vix_df: pd.DataFrame):
        """Set market benchmark data for context features."""
        self.spy_data = spy_df.copy()
        self.vix_data = vix_df.copy()
    
    def detect_regime(self, as_of_date: datetime) -> MarketRegime:
        """
        Detect current market regime using SPY.
        
        Regimes:
        - BULL: SPY > SMA200, positive momentum, ADX > 25
        - BEAR: SPY < SMA200, negative momentum, ADX > 25
        - SIDEWAYS: ADX < 20 or mixed signals
        """
        if self.spy_data is None or len(self.spy_data) < 252:
            return MarketRegime('sideways', 0.5, 'medium', 20.0)
        
        spy = self.spy_data[self.spy_data.index <= as_of_date].copy()
        if len(spy) < 252:
            return MarketRegime('sideways', 0.5, 'medium', 20.0)
        
        close = spy['close'].values.flatten().astype(np.float64)
        high = spy['high'].values.flatten().astype(np.float64)
        low = spy['low'].values.flatten().astype(np.float64)
        
        # Calculate indicators
        sma50 = np.mean(close[-50:])
        sma200 = np.mean(close[-200:])
        current_price = close[-1]
        
        # ADX for trend strength
        adx = calc_adx(high, low, close, 14)
        current_adx = adx[-1] if not np.isnan(adx[-1]) else 20.0
        
        # Momentum (20-day return)
        momentum = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0
        
        # Volatility regime (VIX-based if available)
        if self.vix_data is not None and len(self.vix_data) > 0:
            vix = self.vix_data[self.vix_data.index <= as_of_date]
            if len(vix) > 0:
                vix_close = vix['close']
                if hasattr(vix_close, 'iloc'):
                    current_vix = float(vix_close.iloc[-1])
                else:
                    current_vix = float(vix_close[-1])
                if current_vix < 15:
                    vol_regime = 'low'
                elif current_vix < 25:
                    vol_regime = 'medium'
                else:
                    vol_regime = 'high'
            else:
                vol_regime = 'medium'
        else:
            vol_regime = 'medium'
        
        # Determine regime
        above_sma200 = current_price > sma200
        above_sma50 = current_price > sma50
        strong_trend = current_adx > 25
        
        if strong_trend:
            if above_sma200 and above_sma50 and momentum > 0:
                regime = 'bull'
                strength = min(1.0, current_adx / 40)
            elif not above_sma200 and not above_sma50 and momentum < 0:
                regime = 'bear'
                strength = min(1.0, current_adx / 40)
            else:
                regime = 'sideways'
                strength = 0.5
        else:
            regime = 'sideways'
            strength = 1.0 - (current_adx / 25)
        
        self.current_regime = MarketRegime(regime, strength, vol_regime, current_adx)
        return self.current_regime
    
    def _calculate_symbol_features(self, sym_df: pd.DataFrame, as_of_date: datetime) -> Optional[Dict]:
        """Calculate all features for a single symbol."""
        if len(sym_df) < 252:
            return None
        
        symbol = sym_df['symbol'].iloc[0]
        
        # Arrays for Numba functions
        close = sym_df['close'].values.astype(np.float64)
        high = sym_df['high'].values.astype(np.float64)
        low = sym_df['low'].values.astype(np.float64)
        volume = sym_df['volume'].values.astype(np.float64)
        
        # Prevent division by zero
        volume = np.where(volume == 0, 1, volume)
        
        latest_date = sym_df.index[-1]
        
        # Basic price info
        features = {
            'symbol': symbol,
            'date': latest_date,
            'close': close[-1],
        }
        
        # ===== MOMENTUM FEATURES =====
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[0], returns])
        
        features['return_1d'] = returns[-1]
        features['return_5d'] = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0
        features['return_10d'] = (close[-1] / close[-10] - 1) if len(close) >= 10 else 0
        features['return_20d'] = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0
        features['return_60d'] = (close[-1] / close[-60] - 1) if len(close) >= 60 else 0
        
        # Momentum acceleration
        if len(close) >= 20:
            mom_10d_prev = (close[-11] / close[-20] - 1)
            mom_10d_now = (close[-1] / close[-10] - 1)
            features['momentum_accel'] = mom_10d_now - mom_10d_prev
        else:
            features['momentum_accel'] = 0
        
        # ===== MOVING AVERAGES =====
        sma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
        sma10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
        sma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        sma50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
        sma200 = np.mean(close[-200:]) if len(close) >= 200 else close[-1]
        
        features['sma5_ratio'] = close[-1] / sma5 if sma5 > 0 else 1
        features['sma10_ratio'] = close[-1] / sma10 if sma10 > 0 else 1
        features['sma20_ratio'] = close[-1] / sma20 if sma20 > 0 else 1
        features['sma50_ratio'] = close[-1] / sma50 if sma50 > 0 else 1
        features['sma200_ratio'] = close[-1] / sma200 if sma200 > 0 else 1
        
        # MA crossovers
        features['sma5_above_sma20'] = 1 if sma5 > sma20 else 0
        features['sma20_above_sma50'] = 1 if sma20 > sma50 else 0
        features['sma50_above_sma200'] = 1 if sma50 > sma200 else 0
        
        # EMA
        ema12 = calc_ema(close, 12)
        ema26 = calc_ema(close, 26)
        features['ema12_ratio'] = close[-1] / ema12[-1] if not np.isnan(ema12[-1]) and ema12[-1] > 0 else 1
        features['ema26_ratio'] = close[-1] / ema26[-1] if not np.isnan(ema26[-1]) and ema26[-1] > 0 else 1
        
        # ===== OSCILLATORS =====
        rsi = calc_rsi(close, 14)
        features['rsi_14'] = rsi[-1] if not np.isnan(rsi[-1]) else 50
        features['rsi_overbought'] = 1 if features['rsi_14'] > 70 else 0
        features['rsi_oversold'] = 1 if features['rsi_14'] < 30 else 0
        
        # MACD
        macd_line, signal_line, histogram = calc_macd(close)
        features['macd'] = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
        features['macd_signal'] = signal_line[-1] if not np.isnan(signal_line[-1]) else 0
        features['macd_histogram'] = histogram[-1] if not np.isnan(histogram[-1]) else 0
        features['macd_crossover'] = 1 if features['macd'] > features['macd_signal'] else 0
        
        # Stochastic
        stoch_k, stoch_d = calc_stochastic(high, low, close, 14, 3)
        features['stoch_k'] = stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50
        features['stoch_d'] = stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50
        
        # ===== VOLATILITY =====
        features['volatility_5d'] = np.std(returns[-5:]) * np.sqrt(252) if len(returns) >= 5 else 0
        features['volatility_20d'] = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0
        features['volatility_60d'] = np.std(returns[-60:]) * np.sqrt(252) if len(returns) >= 60 else 0
        
        # Volatility ratio (short vs long)
        if features['volatility_60d'] > 0:
            features['vol_ratio'] = features['volatility_20d'] / features['volatility_60d']
        else:
            features['vol_ratio'] = 1
        
        # ATR
        atr = calc_atr(high, low, close, 14)
        features['atr_14'] = atr[-1] if not np.isnan(atr[-1]) else 0
        features['atr_percent'] = features['atr_14'] / close[-1] if close[-1] > 0 else 0
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calc_bollinger(close, 20, 2.0)
        if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
            bb_width = bb_upper[-1] - bb_lower[-1]
            features['bb_position'] = (close[-1] - bb_lower[-1]) / bb_width if bb_width > 0 else 0.5
            features['bb_width'] = bb_width / bb_middle[-1] if bb_middle[-1] > 0 else 0
        else:
            features['bb_position'] = 0.5
            features['bb_width'] = 0
        
        # ===== TREND STRENGTH =====
        adx = calc_adx(high, low, close, 14)
        features['adx_14'] = adx[-1] if not np.isnan(adx[-1]) else 20
        features['strong_trend'] = 1 if features['adx_14'] > 25 else 0
        
        # ===== VOLUME FEATURES =====
        features['volume_sma20_ratio'] = volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1
        features['volume_sma5_ratio'] = volume[-1] / np.mean(volume[-5:]) if len(volume) >= 5 else 1
        
        # OBV trend
        obv = calc_obv(close, volume)
        obv_sma20 = np.mean(obv[-20:]) if len(obv) >= 20 else obv[-1]
        features['obv_trend'] = 1 if obv[-1] > obv_sma20 else 0
        
        # VWAP
        vwap = calc_vwap(high, low, close, volume)
        features['vwap_ratio'] = close[-1] / vwap[-1] if vwap[-1] > 0 else 1
        
        # ===== PRICE PATTERNS =====
        # 52-week high/low
        high_252 = np.max(high[-252:]) if len(high) >= 252 else high[-1]
        low_252 = np.min(low[-252:]) if len(low) >= 252 else low[-1]
        features['dist_from_52w_high'] = (close[-1] / high_252) - 1
        features['dist_from_52w_low'] = (close[-1] / low_252) - 1
        
        # Recent range position
        high_20 = np.max(high[-20:])
        low_20 = np.min(low[-20:])
        range_20 = high_20 - low_20
        features['range_position_20d'] = (close[-1] - low_20) / range_20 if range_20 > 0 else 0.5
        
        # Gap detection
        if len(close) >= 2:
            features['gap_up'] = 1 if low[-1] > high[-2] else 0
            features['gap_down'] = 1 if high[-1] < low[-2] else 0
        else:
            features['gap_up'] = 0
            features['gap_down'] = 0
        
        # ===== MEAN REVERSION =====
        # Z-score of price
        if len(close) >= 20:
            mean_20 = np.mean(close[-20:])
            std_20 = np.std(close[-20:])
            features['zscore_20d'] = (close[-1] - mean_20) / std_20 if std_20 > 0 else 0
        else:
            features['zscore_20d'] = 0
        
        # Return z-score
        if len(returns) >= 20:
            features['return_zscore'] = (returns[-1] - np.mean(returns[-20:])) / np.std(returns[-20:]) if np.std(returns[-20:]) > 0 else 0
        else:
            features['return_zscore'] = 0
        
        # ===== MARKET CONTEXT =====
        if self.spy_data is not None and len(self.spy_data) > 20:
            spy = self.spy_data[self.spy_data.index <= as_of_date]
            if len(spy) >= 20:
                spy_close = spy['close'].values.flatten()
                spy_returns = np.diff(spy_close) / spy_close[:-1]
                
                # Beta (20-day rolling)
                if len(returns) >= 20 and len(spy_returns) >= 20:
                    stock_rets = returns[-20:]
                    spy_rets = spy_returns[-20:]
                    
                    if len(stock_rets) == len(spy_rets):
                        cov = np.cov(stock_rets, spy_rets)[0, 1]
                        var = np.var(spy_rets)
                        features['beta'] = cov / var if var > 0 else 1
                        
                        # Correlation
                        corr = np.corrcoef(stock_rets, spy_rets)[0, 1]
                        features['corr_to_spy'] = corr if not np.isnan(corr) else 0
                    else:
                        features['beta'] = 1
                        features['corr_to_spy'] = 0
                else:
                    features['beta'] = 1
                    features['corr_to_spy'] = 0
                
                # Relative strength vs SPY
                if len(spy_close) >= 20:
                    stock_ret_20d = features['return_20d']
                    spy_ret_20d = (spy_close[-1] / spy_close[-20] - 1)
                    features['relative_strength'] = stock_ret_20d - spy_ret_20d
                else:
                    features['relative_strength'] = 0
            else:
                features['beta'] = 1
                features['corr_to_spy'] = 0
                features['relative_strength'] = 0
        else:
            features['beta'] = 1
            features['corr_to_spy'] = 0
            features['relative_strength'] = 0
        
        # ===== REGIME FEATURES =====
        if self.current_regime:
            features['regime_bull'] = 1 if self.current_regime.regime == 'bull' else 0
            features['regime_bear'] = 1 if self.current_regime.regime == 'bear' else 0
            features['regime_sideways'] = 1 if self.current_regime.regime == 'sideways' else 0
            features['regime_strength'] = self.current_regime.strength
            features['regime_vol_high'] = 1 if self.current_regime.volatility == 'high' else 0
        else:
            features['regime_bull'] = 0
            features['regime_bear'] = 0
            features['regime_sideways'] = 1
            features['regime_strength'] = 0.5
            features['regime_vol_high'] = 0
        
        return features
    
    def calculate_features(self, df: pd.DataFrame, as_of_date: datetime) -> pd.DataFrame:
        """
        Calculate features for all symbols using parallel processing.
        """
        # Filter to only data available at as_of_date
        df = df[df.index <= as_of_date].copy()
        
        if len(df) < 252:
            return pd.DataFrame()
        
        # Update regime
        self.detect_regime(as_of_date)
        
        # Get unique symbols
        symbols = df['symbol'].unique()
        
        # Process symbols in parallel
        def process_symbol(symbol):
            sym_df = df[df['symbol'] == symbol].sort_index()
            if len(sym_df) >= 200:
                return self._calculate_symbol_features(sym_df, as_of_date)
            return None
        
        # Parallel processing with joblib
        results = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(process_symbol)(symbol) for symbol in symbols
        )
        
        # Filter None results
        features_list = [r for r in results if r is not None]
        
        if not features_list:
            return pd.DataFrame()
        
        return pd.DataFrame(features_list)


# ============================================================================
# ENHANCED WALK-FORWARD BACKTESTER
# ============================================================================

class WalkForwardBacktesterV2:
    """
    Enhanced walk-forward backtesting with:
    - 5-day risk-adjusted return targets
    - GPU-accelerated XGBoost
    - Conviction-weighted position sizing
    - Regime-adaptive behavior
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        train_window_days: int = 252,
        test_window_days: int = 21,
        retrain_frequency: int = 21,
        max_position_pct: float = 0.15,  # Increased for conviction sizing
        top_k: int = 5,
        slippage_bps: float = 10.0,
        commission_per_trade: float = 1.0,
        forward_days: int = 5,  # NEW: Prediction horizon
        use_gpu: bool = True,
    ):
        self.tickers = tickers
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.initial_capital = initial_capital
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.retrain_frequency = retrain_frequency
        self.max_position_pct = max_position_pct
        self.top_k = top_k
        self.slippage_bps = slippage_bps
        self.commission_per_trade = commission_per_trade
        self.forward_days = forward_days
        self.use_gpu = use_gpu
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.regime_history: List[Dict] = []
        
        # Model and features
        self.model = None
        self.feature_calculator = FeatureCalculatorV2()
        self.feature_columns: List[str] = []
        
        # Data storage
        self.price_data: Optional[pd.DataFrame] = None
        self.spy_data: Optional[pd.DataFrame] = None
        
        # Regime tracking for performance
        self.regime_returns = {'bull': [], 'bear': [], 'sideways': []}
        
        # Risk management settings
        self.trailing_stop_pct = 0.08  # 8% trailing stop
        self.hard_stop_pct = 0.10  # 10% hard stop from entry
        self.bear_position_scale = 0.5  # Scale down positions in bear market
        self.bear_max_positions = 2  # Max positions in bear market
        
        # Inverse ETF for hedging
        self.hedge_etf = 'SH'  # ProShares Short S&P 500
        self.hedge_position: Optional[Position] = None
        self.hedge_target_pct = 0.30  # Target 30% hedge in bear markets
        
        # Dynamic universe: ticker first available dates (will be populated during data fetch)
        self.ticker_first_dates: Dict[str, datetime] = {}
        self.min_history_days = train_window_days  # Minimum trading days needed before including ticker
        
        logger.info(f"Initialized WalkForwardBacktesterV2 with GPU={use_gpu}")
    
    def _fetch_data(self) -> None:
        """Fetch all required historical data in parallel."""
        logger.info(f"Fetching data for {len(self.tickers)} tickers using {N_JOBS} threads...")
        
        # Fetch from earliest possible date to capture all crashes
        # We'll filter by availability per ticker later
        data_start = datetime(2007, 1, 1)  # Before 2008 crash
        
        def fetch_ticker(ticker):
            try:
                df = yf.download(
                    ticker,
                    start=data_start.strftime('%Y-%m-%d'),
                    end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    progress=False
                )
                if not df.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.reset_index()
                    df['symbol'] = ticker
                    df.columns = [c.lower() for c in df.columns]
                    df = df.rename(columns={'adj close': 'adj_close'})
                    df = df.set_index('date')
                    return ticker, df
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
            return ticker, None
        
        # Parallel fetch using loky (processes) to avoid yfinance thread-safety issues
        results = Parallel(n_jobs=min(N_JOBS, len(self.tickers)), backend="loky")(
            delayed(fetch_ticker)(ticker) for ticker in self.tickers
        )
        
        all_data = []
        for ticker, df in results:
            if df is not None:
                all_data.append(df)
                # Record first available date for this ticker
                first_date = df.index.min()
                if hasattr(first_date, 'to_pydatetime'):
                    first_date = first_date.to_pydatetime()
                self.ticker_first_dates[ticker] = first_date
                logger.info(f"  {ticker}: data from {first_date.strftime('%Y-%m-%d')} ({len(df)} rows)")
        
        if not all_data:
            raise ValueError("No data fetched for any ticker")
        
        self.price_data = pd.concat(all_data)
        logger.info(f"Fetched {len(self.price_data)} total rows of data")
        
        # Fetch SPY and VIX for regime detection
        for benchmark, attr in [('SPY', 'spy_data'), ('^VIX', 'vix_data')]:
            try:
                df = yf.download(
                    benchmark,
                    start=data_start.strftime('%Y-%m-%d'),
                    end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    progress=False
                )
                # Handle MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                df = df.set_index('date')
                setattr(self, attr, df)
            except Exception as e:
                logger.warning(f"Failed to fetch {benchmark}: {e}")
                setattr(self, attr, pd.DataFrame())
        
        self.feature_calculator.set_market_data(self.spy_data, self.vix_data)
        
        # Log dynamic universe summary
        logger.info("Dynamic Universe Summary:")
        for ticker, first_date in sorted(self.ticker_first_dates.items(), key=lambda x: x[1]):
            logger.info(f"  {ticker}: available from {first_date.strftime('%Y-%m-%d')}")
        
        # Fetch hedge ETF (SH - inverse SPY)
        try:
            hedge_df = yf.download(
                self.hedge_etf,
                start=data_start.strftime('%Y-%m-%d'),
                end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
            if isinstance(hedge_df.columns, pd.MultiIndex):
                hedge_df.columns = hedge_df.columns.get_level_values(0)
            hedge_df = hedge_df.reset_index()
            hedge_df.columns = [c.lower() for c in hedge_df.columns]
            hedge_df['symbol'] = self.hedge_etf
            hedge_df = hedge_df.set_index('date')
            self.hedge_data = hedge_df
            logger.info(f"Fetched {len(hedge_df)} rows for hedge ETF {self.hedge_etf}")
        except Exception as e:
            logger.warning(f"Failed to fetch hedge ETF {self.hedge_etf}: {e}")
            self.hedge_data = pd.DataFrame()
    
    def _get_eligible_tickers(self, as_of_date: datetime) -> List[str]:
        """
        Get tickers that have sufficient history as of the given date.
        
        A ticker is eligible if:
        - It has data (exists in ticker_first_dates)
        - It has at least min_history_days of data before as_of_date
        
        This enables dynamic universe where newer IPOs (IONQ, QBTS) are only
        included once they have enough history for training.
        """
        eligible = []
        for ticker, first_date in self.ticker_first_dates.items():
            # Calculate trading days available (approximate: calendar days * 252/365)
            days_available = (as_of_date - first_date).days
            trading_days_approx = int(days_available * 252 / 365)
            
            if trading_days_approx >= self.min_history_days:
                eligible.append(ticker)
        
        return eligible
    
    def _get_eligible_price_data(self, as_of_date: datetime) -> pd.DataFrame:
        """Get price data filtered to only eligible tickers."""
        eligible = self._get_eligible_tickers(as_of_date)
        if not eligible:
            return pd.DataFrame()
        return self.price_data[self.price_data['symbol'].isin(eligible)]
    
    def _create_training_target(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 5-day forward risk-adjusted return target.
        
        Target = (5-day forward return) / (20-day volatility)
        
        This is a regression target, not binary classification.
        We then convert to probability via ranking.
        """
        df = features_df.copy()
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Calculate forward returns for each symbol
        def calc_forward(group):
            group = group.sort_values('date')
            # 5-day forward return
            group['forward_return'] = group['close'].shift(-self.forward_days) / group['close'] - 1
            return group
        
        df = df.groupby('symbol', group_keys=False).apply(calc_forward)
        
        # Risk-adjusted target: forward return / recent volatility
        df['target'] = df['forward_return'] / (df['volatility_20d'] + 0.001)
        
        # Clip extreme values
        df['target'] = df['target'].clip(-3, 3)
        
        # Drop rows without target
        df = df.dropna(subset=['target'])
        
        return df
    
    def _train_model(self, as_of_date: datetime) -> None:
        """Train XGBoost model with GPU acceleration."""
        # Get eligible tickers for this date
        eligible_tickers = self._get_eligible_tickers(as_of_date)
        if not eligible_tickers:
            logger.warning(f"No eligible tickers as of {as_of_date.strftime('%Y-%m-%d')}")
            return
        
        logger.info(f"Training model as of {as_of_date.strftime('%Y-%m-%d')} with {len(eligible_tickers)} tickers: {eligible_tickers}")
        
        # Get price data filtered to eligible tickers
        eligible_data = self._get_eligible_price_data(as_of_date)
        if eligible_data.empty:
            logger.warning("No eligible price data for training")
            return
        
        # Get features for training period
        train_start = as_of_date - timedelta(days=self.train_window_days)
        
        # Sample every N days instead of every day to reduce compute
        # but still capture the full training window
        sample_interval = 3  # Every 3 trading days
        
        features_list = []
        current = train_start
        day_count = 0
        
        while current <= as_of_date - timedelta(days=self.forward_days):  # Leave room for forward returns
            if current.weekday() < 5:
                day_count += 1
                if day_count % sample_interval == 0:
                    day_features = self.feature_calculator.calculate_features(
                        eligible_data, current
                    )
                    if not day_features.empty:
                        features_list.append(day_features)
            current += timedelta(days=1)
        
        if not features_list:
            logger.warning("No features available for training")
            return
        
        features_df = pd.concat(features_list, ignore_index=True)
        
        # Create target
        features_df = self._create_training_target(features_df)
        
        if len(features_df) < 100:
            logger.warning(f"Insufficient training data: {len(features_df)} rows")
            return
        
        # Feature columns
        exclude_cols = ['symbol', 'date', 'close', 'target', 'forward_return']
        self.feature_columns = [c for c in features_df.columns if c not in exclude_cols]
        
        X = features_df[self.feature_columns].fillna(0).values
        y = features_df['target'].values
        
        # XGBoost with GPU
        if self.use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'
        
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            device=device,
            n_jobs=N_JOBS,
            random_state=42,
            verbosity=0,
        )
        
        self.model.fit(X, y)
        logger.info(f"Model trained on {len(X)} samples using {device.upper()}")
    
    def _get_signals(self, as_of_date: datetime) -> Dict[str, Tuple[float, float]]:
        """
        Generate signals with conviction scores.
        
        Returns: {symbol: (signal, conviction)}
        - signal: predicted risk-adjusted return
        - conviction: 0-1 score based on prediction confidence
        """
        if self.model is None:
            return {}
        
        # Only generate signals for eligible tickers
        eligible_data = self._get_eligible_price_data(as_of_date)
        if eligible_data.empty:
            return {}
        
        features = self.feature_calculator.calculate_features(eligible_data, as_of_date)
        
        if features.empty:
            return {}
        
        # Ensure all columns exist
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        X = features[self.feature_columns].fillna(0).values
        
        # Get predictions
        try:
            predictions = self.model.predict(X)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return {}
        
        # Convert to conviction scores (normalize to 0-1)
        pred_min = predictions.min()
        pred_max = predictions.max()
        
        if pred_max - pred_min > 0:
            normalized = (predictions - pred_min) / (pred_max - pred_min)
        else:
            normalized = np.ones_like(predictions) * 0.5
        
        signals = {}
        for i, row in features.iterrows():
            signals[row['symbol']] = (predictions[i], normalized[i])
        
        return signals
    
    def _get_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get closing price for symbol on date."""
        try:
            sym_data = self.price_data[self.price_data['symbol'] == symbol]
            date_data = sym_data[sym_data.index == date]
            if not date_data.empty:
                return float(date_data['close'].iloc[0])
            
            # Try nearby dates
            nearby = sym_data[(sym_data.index >= date - timedelta(days=3)) & 
                            (sym_data.index <= date)]
            if not nearby.empty:
                return float(nearby['close'].iloc[-1])
        except Exception:
            pass
        return None
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        slip = self.slippage_bps / 10000
        if side == 'buy':
            return price * (1 + slip)
        return price * (1 - slip)
    
    def _get_hedge_price(self, date: datetime) -> Optional[float]:
        """Get price for hedge ETF on date."""
        if not hasattr(self, 'hedge_data') or self.hedge_data.empty:
            return None
        try:
            date_data = self.hedge_data[self.hedge_data.index == date]
            if not date_data.empty:
                return float(date_data['close'].iloc[0])
            # Try nearby dates
            nearby = self.hedge_data[
                (self.hedge_data.index >= date - timedelta(days=3)) & 
                (self.hedge_data.index <= date)
            ]
            if not nearby.empty:
                return float(nearby['close'].iloc[-1])
        except Exception:
            pass
        return None
    
    def _manage_hedge_position(self, date: datetime, regime: MarketRegime) -> None:
        """Manage hedge position based on market regime."""
        hedge_price = self._get_hedge_price(date)
        if hedge_price is None:
            return
        
        portfolio_value = self._get_portfolio_value(date)
        
        if regime.regime == 'bear':
            # In bear market, maintain hedge position
            target_hedge_value = portfolio_value * self.hedge_target_pct
            
            if self.hedge_position is None:
                # Open hedge position
                shares = int(target_hedge_value / hedge_price)
                if shares > 0:
                    exec_price = self._apply_slippage(hedge_price, 'buy')
                    cost = shares * exec_price + self.commission_per_trade
                    if cost <= self.cash:
                        self.cash -= cost
                        self.hedge_position = Position(
                            symbol=self.hedge_etf,
                            shares=shares,
                            entry_price=exec_price,
                            entry_date=date,
                            conviction=1.0,
                            side='hedge'
                        )
                        logger.debug(f"HEDGE BUY {shares} {self.hedge_etf} @ ${exec_price:.2f}")
            else:
                # Adjust hedge if needed (rebalance if off by >20%)
                current_value = self.hedge_position.shares * hedge_price
                if current_value < target_hedge_value * 0.8:
                    # Add to hedge
                    add_value = target_hedge_value - current_value
                    add_shares = int(add_value / hedge_price)
                    if add_shares > 0:
                        exec_price = self._apply_slippage(hedge_price, 'buy')
                        cost = add_shares * exec_price + self.commission_per_trade
                        if cost <= self.cash:
                            self.cash -= cost
                            self.hedge_position.shares += add_shares
                            logger.debug(f"HEDGE ADD {add_shares} {self.hedge_etf} @ ${exec_price:.2f}")
        else:
            # Not in bear market, close hedge if exists
            if self.hedge_position is not None:
                exec_price = self._apply_slippage(hedge_price, 'sell')
                proceeds = self.hedge_position.shares * exec_price - self.commission_per_trade
                pnl = proceeds - self.hedge_position.cost_basis
                
                # Record as trade
                self.trades.append(Trade(
                    symbol=self.hedge_etf,
                    side='hedge',
                    shares=self.hedge_position.shares,
                    entry_price=self.hedge_position.entry_price,
                    exit_price=exec_price,
                    entry_date=self.hedge_position.entry_date,
                    exit_date=date,
                    pnl=pnl,
                    pnl_percent=pnl / self.hedge_position.cost_basis,
                    holding_days=(date - self.hedge_position.entry_date).days,
                    conviction=1.0
                ))
                
                self.cash += proceeds
                logger.debug(f"HEDGE CLOSE {self.hedge_position.shares} {self.hedge_etf} @ ${exec_price:.2f}, P&L: ${pnl:.2f}")
                self.hedge_position = None
    
    def _calculate_position_size(self, conviction: float) -> float:
        """
        Calculate position size based on conviction.
        
        Higher conviction = larger position (up to max_position_pct)
        Lower conviction = smaller position
        """
        # Base size is half of max
        base_pct = self.max_position_pct / 2
        
        # Scale by conviction (0.5 to 1.0 of conviction range maps to base to max)
        if conviction > 0.7:
            size_pct = base_pct + (conviction - 0.5) * (self.max_position_pct - base_pct) / 0.5
        else:
            size_pct = base_pct * conviction / 0.7
        
        return min(size_pct, self.max_position_pct)
    
    def _execute_buy(self, symbol: str, price: float, date: datetime, conviction: float) -> bool:
        """Execute a buy order with conviction-based sizing."""
        if symbol in self.positions:
            return False
        
        exec_price = self._apply_slippage(price, 'buy')
        
        # Conviction-based position sizing
        position_pct = self._calculate_position_size(conviction)
        max_value = self.cash * position_pct
        
        shares = int(max_value / exec_price)
        
        if shares <= 0:
            return False
        
        cost = shares * exec_price + self.commission_per_trade
        
        if cost > self.cash:
            shares = int((self.cash - self.commission_per_trade) / exec_price)
            if shares <= 0:
                return False
            cost = shares * exec_price + self.commission_per_trade
        
        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=exec_price,
            entry_date=date,
            conviction=conviction
        )
        
        logger.debug(f"BUY {shares} {symbol} @ ${exec_price:.2f} (conviction: {conviction:.2f})")
        return True
    
    def _execute_sell(self, symbol: str, price: float, date: datetime) -> bool:
        """Execute a sell order."""
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        exec_price = self._apply_slippage(price, 'sell')
        
        proceeds = pos.shares * exec_price - self.commission_per_trade
        pnl = proceeds - pos.cost_basis
        pnl_pct = pnl / pos.cost_basis
        
        holding_days = (date - pos.entry_date).days
        
        self.trades.append(Trade(
            symbol=symbol,
            side='long',
            shares=pos.shares,
            entry_price=pos.entry_price,
            exit_price=exec_price,
            entry_date=pos.entry_date,
            exit_date=date,
            pnl=pnl,
            pnl_percent=pnl_pct,
            holding_days=holding_days,
            conviction=pos.conviction
        ))
        
        # Track regime performance
        if self.feature_calculator.current_regime:
            regime = self.feature_calculator.current_regime.regime
            self.regime_returns[regime].append(pnl_pct)
        
        self.cash += proceeds
        del self.positions[symbol]
        
        logger.debug(f"SELL {pos.shares} {symbol} @ ${exec_price:.2f}, P&L: ${pnl:.2f}")
        return True
    
    def _get_portfolio_value(self, date: datetime) -> float:
        """Calculate total portfolio value."""
        positions_value = 0
        for symbol, pos in self.positions.items():
            price = self._get_price(symbol, date)
            if price:
                positions_value += pos.shares * price
            else:
                positions_value += pos.cost_basis
        
        # Include hedge position
        if self.hedge_position is not None:
            hedge_price = self._get_hedge_price(date)
            if hedge_price:
                positions_value += self.hedge_position.shares * hedge_price
            else:
                positions_value += self.hedge_position.cost_basis
        
        return self.cash + positions_value
    
    def run(self) -> WalkForwardResult:
        """Run the enhanced walk-forward backtest."""
        logger.info(f"Starting walk-forward backtest V2: {self.start_date} to {self.end_date}")
        
        # Fetch all data
        self._fetch_data()
        
        # Initialize
        current_date = self.start_date
        last_train_date = None
        n_train_periods = 0
        trading_days = []
        
        # Regime-adaptive thresholds
        def get_thresholds(regime: MarketRegime):
            """Adjust thresholds based on regime."""
            if regime.regime == 'bull':
                return 0.55, 0.35, 0.85  # buy, sell, hold_if_above
            elif regime.regime == 'bear':
                return 0.85, 0.50, 0.95  # MUCH more conservative in bear - barely trade
            else:
                return 0.60, 0.40, 0.85  # Sideways: standard
        
        # Main simulation loop
        while current_date <= self.end_date:
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            trading_days.append(current_date)
            
            # Retrain model if needed
            if (last_train_date is None or 
                (current_date - last_train_date).days >= self.retrain_frequency):
                self._train_model(current_date)
                last_train_date = current_date
                n_train_periods += 1
            
            # Detect current regime
            regime = self.feature_calculator.detect_regime(current_date)
            self.regime_history.append({
                'date': current_date.strftime('%Y-%m-%d'),
                **regime.to_dict()
            })
            
            # Manage hedge position based on regime
            self._manage_hedge_position(current_date, regime)
            
            # Get regime-adjusted thresholds
            buy_thresh, sell_thresh, hold_thresh = get_thresholds(regime)
            
            # Determine max positions for current regime
            max_positions = self.bear_max_positions if regime.regime == 'bear' else self.top_k
            
            # Get signals
            signals = self._get_signals(current_date)
            
            if signals:
                # Sort by predicted return (first element of tuple)
                sorted_signals = sorted(signals.items(), key=lambda x: x[1][0], reverse=True)
                
                # Close positions - check trailing stops and other exit conditions
                for symbol in list(self.positions.keys()):
                    if symbol in signals:
                        pred, conviction = signals[symbol]
                    else:
                        pred, conviction = 0, 0
                    
                    price = self._get_price(symbol, current_date)
                    pos = self.positions[symbol]
                    holding_days = (current_date - pos.entry_date).days
                    
                    should_sell = False
                    sell_reason = ""
                    
                    if price:
                        # Update high water mark for trailing stop
                        pos.update_high_water_mark(price)
                        
                        # Check trailing stop (8% from high)
                        if pos.check_trailing_stop(price, self.trailing_stop_pct):
                            should_sell = True
                            sell_reason = "trailing_stop"
                        # Check hard stop (10% from entry)
                        elif (price / pos.entry_price - 1) < -self.hard_stop_pct:
                            should_sell = True
                            sell_reason = "hard_stop"
                        # Sell if conviction dropped
                        elif conviction < sell_thresh:
                            should_sell = True
                            sell_reason = "low_conviction"
                        # Held for target horizon - take profit or cut loss
                        elif holding_days >= self.forward_days:
                            should_sell = True
                            sell_reason = "horizon_reached"
                        # Force reduce positions in bear market if we have too many
                        elif regime.regime == 'bear' and len(self.positions) > max_positions:
                            should_sell = True
                            sell_reason = "bear_reduction"
                    
                    if should_sell and price:
                        logger.debug(f"EXIT {symbol}: {sell_reason}")
                        self._execute_sell(symbol, price, current_date)
                
                # Open new positions - prioritize by predicted return
                # In bear market, scale down position sizes and limit count
                open_slots = max_positions - len(self.positions)
                
                for symbol, (pred, conviction) in sorted_signals[:open_slots * 2]:
                    if len(self.positions) >= max_positions:
                        break
                    
                    if conviction >= buy_thresh and symbol not in self.positions:
                        price = self._get_price(symbol, current_date)
                        if price:
                            # Scale down conviction in bear market for smaller positions
                            adj_conviction = conviction * self.bear_position_scale if regime.regime == 'bear' else conviction
                            self._execute_buy(symbol, price, current_date, adj_conviction)
            
            # Record equity
            portfolio_value = self._get_portfolio_value(current_date)
            self.equity_curve.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': self.cash,
                'n_positions': len(self.positions),
                'regime': regime.regime
            })
            
            current_date += timedelta(days=1)
        
        # Close all positions at end
        final_date = trading_days[-1] if trading_days else self.end_date
        for symbol in list(self.positions.keys()):
            price = self._get_price(symbol, final_date)
            if price:
                self._execute_sell(symbol, price, final_date)
        
        # Close hedge position at end
        if self.hedge_position is not None:
            hedge_price = self._get_hedge_price(final_date)
            if hedge_price:
                exec_price = self._apply_slippage(hedge_price, 'sell')
                proceeds = self.hedge_position.shares * exec_price - self.commission_per_trade
                pnl = proceeds - self.hedge_position.cost_basis
                self.trades.append(Trade(
                    symbol=self.hedge_etf,
                    side='hedge',
                    shares=self.hedge_position.shares,
                    entry_price=self.hedge_position.entry_price,
                    exit_price=exec_price,
                    entry_date=self.hedge_position.entry_date,
                    exit_date=final_date,
                    pnl=pnl,
                    pnl_percent=pnl / self.hedge_position.cost_basis,
                    holding_days=(final_date - self.hedge_position.entry_date).days,
                    conviction=1.0
                ))
                self.cash += proceeds
                self.hedge_position = None
        
        # Calculate results
        return self._calculate_results(n_train_periods)

    def _calculate_results(self, n_train_periods: int) -> WalkForwardResult:
        """Calculate comprehensive backtest results."""
        equity_df = pd.DataFrame(self.equity_curve)
        
        if equity_df.empty:
            raise ValueError("No equity curve data")
        
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df = equity_df.set_index('date')
        
        final_value = equity_df['value'].iloc[-1]
        
        # Calculate returns
        equity_df['returns'] = equity_df['value'].pct_change()
        total_return = final_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Annualized return
        days = (self.end_date - self.start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0
        
        # Benchmark returns
        if self.spy_data is not None and len(self.spy_data) > 0:
            spy_start_data = self.spy_data[self.spy_data.index >= self.start_date]
            spy_end_data = self.spy_data[self.spy_data.index <= self.end_date]
            
            if len(spy_start_data) > 0 and len(spy_end_data) > 0:
                # Handle potential MultiIndex - flatten to get scalar values
                spy_start_val = spy_start_data['close'].iloc[0]
                if hasattr(spy_start_val, 'values'):
                    spy_start_val = spy_start_val.values.flatten()[0]
                spy_start = float(spy_start_val)
                
                spy_end_val = spy_end_data['close'].iloc[-1]
                if hasattr(spy_end_val, 'values'):
                    spy_end_val = spy_end_val.values.flatten()[0]
                spy_end = float(spy_end_val)
                
                benchmark_return = spy_end - spy_start
                benchmark_return_pct = (spy_end / spy_start) - 1
                
                benchmark_df = self.spy_data[
                    (self.spy_data.index >= self.start_date) & 
                    (self.spy_data.index <= self.end_date)
                ].copy()
                # Get close as a 1D series
                close_col = benchmark_df['close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                benchmark_df = pd.DataFrame({'close': close_col})
                benchmark_df['value'] = self.initial_capital * (benchmark_df['close'] / spy_start)
            else:
                benchmark_return = 0
                benchmark_return_pct = 0
                benchmark_df = pd.DataFrame()
        else:
            benchmark_return = 0
            benchmark_return_pct = 0
            benchmark_df = pd.DataFrame()
        
        # Alpha (annualized)
        benchmark_ann = (1 + benchmark_return_pct) ** (1 / years) - 1 if years > 0 else 0
        alpha = annualized_return - benchmark_ann
        
        # Risk metrics
        daily_returns = equity_df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
        sortino = (daily_returns.mean() * 252) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        equity_df['peak'] = equity_df['value'].cummax()
        equity_df['drawdown'] = equity_df['value'] - equity_df['peak']
        equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak']
        
        max_dd = abs(equity_df['drawdown'].min())
        max_dd_pct = abs(equity_df['drawdown_pct'].min())
        
        # Calmar ratio
        calmar = annualized_return / max_dd_pct if max_dd_pct > 0 else 0
        
        # Max drawdown duration
        in_drawdown = equity_df['drawdown'] < 0
        if in_drawdown.any():
            dd_groups = (~in_drawdown).cumsum()
            dd_durations = in_drawdown.groupby(dd_groups).sum()
            max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0
        else:
            max_dd_duration = 0
        
        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in self.trades if t.pnl <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float('inf')
        
        avg_holding = np.mean([t.holding_days for t in self.trades]) if self.trades else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Statistical significance
        if len(daily_returns) > 30 and len(benchmark_df) > 0:
            benchmark_returns = benchmark_df['close'].pct_change().dropna()
            common_dates = daily_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 30:
                strat_rets = daily_returns.loc[common_dates]
                bench_rets = benchmark_returns.loc[common_dates]
                excess_returns = strat_rets.values - bench_rets.values
                t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
            else:
                t_stat, p_value = 0, 1
        else:
            t_stat, p_value = 0, 1
        
        # Regime returns
        bull_return = np.mean(self.regime_returns['bull']) if self.regime_returns['bull'] else 0
        bear_return = np.mean(self.regime_returns['bear']) if self.regime_returns['bear'] else 0
        sideways_return = np.mean(self.regime_returns['sideways']) if self.regime_returns['sideways'] else 0
        
        metrics = BacktestMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            benchmark_return_pct=benchmark_return_pct,
            alpha=alpha,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            downside_volatility=downside_vol,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_period=avg_holding,
            expectancy=expectancy,
            t_statistic=t_stat,
            p_value=p_value,
            bull_return=bull_return,
            bear_return=bear_return,
            sideways_return=sideways_return,
        )
        
        return WalkForwardResult(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            equity_curve=equity_df,
            benchmark_curve=benchmark_df,
            drawdown_curve=equity_df[['drawdown', 'drawdown_pct']],
            metrics=metrics,
            trades=self.trades,
            n_train_periods=n_train_periods,
            train_window_days=self.train_window_days,
            test_window_days=self.test_window_days,
            regime_history=self.regime_history,
        )


# ============================================================================
# HTML REPORT GENERATOR (ENHANCED)
# ============================================================================

def generate_backtest_report_v2(result: WalkForwardResult, output_path: str = None) -> str:
    """Generate an enhanced HTML report for backtest results."""
    
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'backtest_report_v2.html'
        )
    
    m = result.metrics
    
    # Prepare data for charts
    equity_dates = result.equity_curve.index.strftime('%Y-%m-%d').tolist()
    equity_values = result.equity_curve['value'].tolist()
    
    if not result.benchmark_curve.empty:
        bench_dates = result.benchmark_curve.index.strftime('%Y-%m-%d').tolist()
        bench_values = result.benchmark_curve['value'].tolist()
    else:
        bench_dates = equity_dates
        bench_values = [result.initial_capital] * len(equity_dates)
    
    dd_values = (result.drawdown_curve['drawdown_pct'] * 100).tolist()
    
    # Trade list (last 100)
    trades_html = ''
    for t in result.trades[-100:]:
        pnl_class = 'positive' if t.pnl > 0 else 'negative'
        trades_html += f'''
        <tr>
            <td>{t.entry_date.strftime('%Y-%m-%d')}</td>
            <td>{t.exit_date.strftime('%Y-%m-%d')}</td>
            <td><strong>{t.symbol}</strong></td>
            <td>{t.shares}</td>
            <td>${t.entry_price:.2f}</td>
            <td>${t.exit_price:.2f}</td>
            <td class="{pnl_class}">${t.pnl:+,.2f}</td>
            <td class="{pnl_class}">{t.pnl_percent:+.2%}</td>
            <td>{t.holding_days}d</td>
            <td>{t.conviction:.2f}</td>
        </tr>
        '''
    
    # Regime breakdown
    regime_counts = {'bull': 0, 'bear': 0, 'sideways': 0}
    for r in result.regime_history:
        regime_counts[r['regime']] = regime_counts.get(r['regime'], 0) + 1
    total_days = sum(regime_counts.values())
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Walk-Forward Backtest Report V2</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #22c55e, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ text-align: center; color: #71717a; margin-bottom: 30px; font-size: 0.9rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px; }}
        .card {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
        }}
        .card h2 {{ font-size: 0.75rem; color: #71717a; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }}
        .metric {{ font-size: 2.2rem; font-weight: 700; }}
        .metric.positive {{ color: #22c55e; }}
        .metric.negative {{ color: #ef4444; }}
        .metric.neutral {{ color: #f59e0b; }}
        .metric-label {{ color: #71717a; font-size: 0.8rem; margin-top: 4px; }}
        .full-width {{ grid-column: 1 / -1; }}
        .chart-container {{ height: 350px; position: relative; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
        th, td {{ padding: 12px 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.06); }}
        th {{ color: #71717a; font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px; }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .warning {{ 
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05)); 
            padding: 20px; 
            border-radius: 12px; 
            margin-bottom: 20px; 
            border-left: 4px solid #f59e0b;
        }}
        .success {{ 
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05)); 
            padding: 20px; 
            border-radius: 12px; 
            margin-bottom: 20px; 
            border-left: 4px solid #22c55e;
        }}
        .stats-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
        .stat {{ 
            flex: 1; 
            min-width: 80px; 
            text-align: center; 
            padding: 16px 8px; 
            background: rgba(255,255,255,0.02); 
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .stat-value {{ font-size: 1.4rem; font-weight: 600; }}
        .stat-label {{ font-size: 0.65rem; color: #71717a; text-transform: uppercase; margin-top: 4px; letter-spacing: 0.5px; }}
        .regime-bar {{
            display: flex;
            height: 24px;
            border-radius: 6px;
            overflow: hidden;
            margin-top: 12px;
        }}
        .regime-bull {{ background: #22c55e; }}
        .regime-bear {{ background: #ef4444; }}
        .regime-sideways {{ background: #71717a; }}
        .hardware-badge {{
            display: inline-block;
            background: rgba(139, 92, 246, 0.2);
            color: #a78bfa;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            margin: 0 4px;
        }}
        @media (max-width: 900px) {{
            .grid-3 {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Walk-Forward Backtest Report V2</h1>
        <p class="subtitle">
            {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')} | 
            {result.n_train_periods} training periods |
            <span class="hardware-badge">XGBoost + CUDA</span>
            <span class="hardware-badge">Numba JIT</span>
            <span class="hardware-badge">{N_JOBS} CPU Threads</span>
        </p>
        
        {'<div class="warning"><strong>⚠️ UNDERPERFORMED BENCHMARK:</strong> Strategy returned ' + f'{m.total_return_pct:+.2%}' + ' vs SPY ' + f'{m.benchmark_return_pct:+.2%}' + '. Alpha: ' + f'{m.alpha:+.2%}' + '. P-value: ' + f'{m.p_value:.4f}' + ' (not statistically significant).</div>' if m.alpha < 0 or m.p_value > 0.05 else '<div class="success"><strong>✓ POSITIVE ALPHA:</strong> Strategy outperformed benchmark by ' + f'{m.alpha:+.2%}' + ' (annualized). P-value: ' + f'{m.p_value:.4f}' + '</div>'}
        
        <div class="grid">
            <div class="card">
                <h2>Total Return</h2>
                <div class="metric {'positive' if m.total_return >= 0 else 'negative'}">{m.total_return_pct:+.2%}</div>
                <div class="metric-label">${m.total_return:+,.2f} | ${result.initial_capital:,.0f} → ${result.final_value:,.0f}</div>
            </div>
            <div class="card">
                <h2>Benchmark (SPY)</h2>
                <div class="metric {'positive' if m.benchmark_return_pct >= 0 else 'negative'}">{m.benchmark_return_pct:+.2%}</div>
                <div class="metric-label">Buy & Hold</div>
            </div>
            <div class="card">
                <h2>Alpha (Annualized)</h2>
                <div class="metric {'positive' if m.alpha >= 0 else 'negative'}">{m.alpha:+.2%}</div>
                <div class="metric-label">Excess Return vs SPY</div>
            </div>
            <div class="card">
                <h2>Sharpe Ratio</h2>
                <div class="metric {'positive' if m.sharpe_ratio >= 1 else 'neutral' if m.sharpe_ratio >= 0.5 else 'negative'}">{m.sharpe_ratio:.2f}</div>
                <div class="metric-label">Risk-Adjusted Return</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card full-width">
                <h2>Equity Curve vs Benchmark</h2>
                <div class="chart-container">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card full-width">
                <h2>Drawdown</h2>
                <div class="chart-container" style="height: 200px;">
                    <canvas id="drawdownChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="grid-3">
            <div class="card">
                <h2>Risk Metrics</h2>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-value">{m.sortino_ratio:.2f}</div>
                        <div class="stat-label">Sortino</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{m.calmar_ratio:.2f}</div>
                        <div class="stat-label">Calmar</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value negative">{m.max_drawdown_pct:.1%}</div>
                        <div class="stat-label">Max DD</div>
                    </div>
                </div>
                <div class="stats-row" style="margin-top: 12px;">
                    <div class="stat">
                        <div class="stat-value">{m.volatility:.1%}</div>
                        <div class="stat-label">Volatility</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{m.max_drawdown_duration}d</div>
                        <div class="stat-label">DD Duration</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>Trade Statistics</h2>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-value">{m.total_trades}</div>
                        <div class="stat-label">Trades</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{m.win_rate:.1%}</div>
                        <div class="stat-label">Win Rate</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{m.profit_factor:.2f}</div>
                        <div class="stat-label">Profit Factor</div>
                    </div>
                </div>
                <div class="stats-row" style="margin-top: 12px;">
                    <div class="stat">
                        <div class="stat-value positive">${m.avg_win:.0f}</div>
                        <div class="stat-label">Avg Win</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value negative">${m.avg_loss:.0f}</div>
                        <div class="stat-label">Avg Loss</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{m.avg_holding_period:.1f}d</div>
                        <div class="stat-label">Avg Hold</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>Market Regime Performance</h2>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-value {'positive' if m.bull_return >= 0 else 'negative'}">{m.bull_return:+.2%}</div>
                        <div class="stat-label">Bull Avg</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value {'positive' if m.bear_return >= 0 else 'negative'}">{m.bear_return:+.2%}</div>
                        <div class="stat-label">Bear Avg</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value {'positive' if m.sideways_return >= 0 else 'negative'}">{m.sideways_return:+.2%}</div>
                        <div class="stat-label">Sideways Avg</div>
                    </div>
                </div>
                <div class="regime-bar">
                    <div class="regime-bull" style="width: {regime_counts['bull']/total_days*100 if total_days > 0 else 0}%"></div>
                    <div class="regime-sideways" style="width: {regime_counts['sideways']/total_days*100 if total_days > 0 else 0}%"></div>
                    <div class="regime-bear" style="width: {regime_counts['bear']/total_days*100 if total_days > 0 else 0}%"></div>
                </div>
                <div class="metric-label" style="margin-top: 8px;">
                    Bull: {regime_counts['bull']}d | Sideways: {regime_counts['sideways']}d | Bear: {regime_counts['bear']}d
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card full-width">
                <h2>Expectancy & Edge</h2>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-value {'positive' if m.expectancy >= 0 else 'negative'}">${m.expectancy:+,.2f}</div>
                        <div class="stat-label">Expected $ Per Trade</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{m.t_statistic:.2f}</div>
                        <div class="stat-label">T-Statistic</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value {'positive' if m.p_value < 0.05 else 'negative'}">{m.p_value:.4f}</div>
                        <div class="stat-label">P-Value</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{'YES ✓' if m.p_value < 0.05 else 'NO'}</div>
                        <div class="stat-label">Significant (p<0.05)</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card full-width">
                <h2>Recent Trades (Last 100)</h2>
                <div style="overflow-x: auto; max-height: 500px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>Symbol</th>
                                <th>Shares</th>
                                <th>Entry $</th>
                                <th>Exit $</th>
                                <th>P&L</th>
                                <th>P&L %</th>
                                <th>Hold</th>
                                <th>Conviction</th>
                            </tr>
                        </thead>
                        <tbody>
                            {trades_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ labels: {{ color: '#e4e4e7' }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#71717a', maxTicksLimit: 12 }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }},
                y: {{ ticks: {{ color: '#71717a' }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }}
            }}
        }};
        
        // Equity Chart
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(equity_dates)},
                datasets: [{{
                    label: 'Strategy',
                    data: {json.dumps(equity_values)},
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 2
                }}, {{
                    label: 'SPY (Benchmark)',
                    data: {json.dumps(bench_values)},
                    borderColor: '#71717a',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 1.5
                }}]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{ ...chartOptions.scales.y, ticks: {{ ...chartOptions.scales.y.ticks, callback: v => '$' + v.toLocaleString() }} }}
                }}
            }}
        }});
        
        // Drawdown Chart
        new Chart(document.getElementById('drawdownChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(equity_dates)},
                datasets: [{{
                    label: 'Drawdown %',
                    data: {json.dumps(dd_values)},
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.2)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }}]
            }},
            options: {{
                ...chartOptions,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    ...chartOptions.scales,
                    y: {{ ...chartOptions.scales.y, ticks: {{ ...chartOptions.scales.y.ticks, callback: v => v.toFixed(1) + '%' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Backtest report V2 saved to {output_path}")
    return output_path


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_walkforward_backtest_v2(
    tickers: List[str],
    start_date: str,
    end_date: str,
    **kwargs
) -> WalkForwardResult:
    """
    Run an enhanced walk-forward backtest.
    """
    backtester = WalkForwardBacktesterV2(tickers, start_date, end_date, **kwargs)
    result = backtester.run()
    
    # Generate HTML report
    generate_backtest_report_v2(result)
    
    return result


if __name__ == '__main__':
    # Example run
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    
    result = run_walkforward_backtest_v2(
        tickers=tickers,
        start_date='2023-01-01',
        end_date='2024-12-01',
        initial_capital=100000,
        use_gpu=True,
    )
    
    print(result.summary())
