"""
Walk-Forward Backtesting Engine for quantSim

This module implements a rigorous walk-forward backtesting framework that:

1. PREVENTS LOOK-AHEAD BIAS
   - Never uses future data for training or feature calculation
   - Recalculates all features using only data available at each point in time
   - Retrains model periodically using only historical data

2. REALISTIC EXECUTION
   - Slippage modeling (market impact)
   - Transaction costs (commissions + spread)
   - Position sizing with risk limits
   - Liquidity constraints

3. PROPER VALIDATION
   - Walk-forward: train on [t-N, t], test on [t, t+M], slide window
   - Multiple train/test periods to assess robustness
   - Out-of-sample performance only (never in-sample)

4. BENCHMARK COMPARISON
   - Compare against buy-and-hold SPY
   - Risk-adjusted returns (Sharpe, Sortino)
   - Statistical significance testing

Architecture:
    [Historical Data] --> [Feature Builder] --> [Model Training]
                              |                       |
                              v                       v
    [Current Data] -------> [Features] --------> [Predictions] --> [Signals]
                                                                       |
                                                                       v
                                                               [Execution Sim]
                                                                       |
                                                                       v
                                                               [Portfolio State]
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from scipy import stats
import json
import os

from src.logging import get_logger

logger = get_logger(__name__)


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
    side: str = 'long'
    
    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price


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
        }


@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest."""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    benchmark_return: float
    benchmark_return_pct: float
    alpha: float  # Excess return over benchmark
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int  # days
    volatility: float  # annualized
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
    
    # Statistical tests
    t_statistic: float  # vs benchmark
    p_value: float  # statistical significance
    
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
    
    def summary(self) -> str:
        """Return formatted summary."""
        m = self.metrics
        return f"""
================================================================================
                    WALK-FORWARD BACKTEST RESULTS
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
Max Drawdown:       ${m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2%})
Max DD Duration:    {m.max_drawdown_duration} days
Volatility (ann.):  {m.volatility:.2%}

--- TRADE STATISTICS ---
Total Trades:       {m.total_trades}
Win Rate:           {m.win_rate:.2%}
Avg Win:            ${m.avg_win:,.2f}
Avg Loss:           ${m.avg_loss:,.2f}
Profit Factor:      {m.profit_factor:.2f}
Avg Holding Period: {m.avg_holding_period:.1f} days

--- STATISTICAL SIGNIFICANCE ---
T-Statistic vs SPY: {m.t_statistic:.2f}
P-Value:            {m.p_value:.4f}
Significant (p<0.05): {'YES' if m.p_value < 0.05 else 'NO'}

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
# FEATURE CALCULATOR (Standalone, no look-ahead)
# ============================================================================

class FeatureCalculator:
    """
    Calculates features using only data available at each point in time.
    This is critical for preventing look-ahead bias.
    """
    
    def __init__(self):
        self.spy_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None
    
    def set_market_data(self, spy_df: pd.DataFrame, vix_df: pd.DataFrame):
        """Set market benchmark data for context features."""
        self.spy_data = spy_df.copy()
        self.vix_data = vix_df.copy()
    
    def calculate_features(self, df: pd.DataFrame, as_of_date: datetime) -> pd.DataFrame:
        """
        Calculate features for all symbols using only data up to as_of_date.
        
        CRITICAL: This function must NEVER use data after as_of_date.
        """
        # Filter to only data available at as_of_date
        df = df[df.index <= as_of_date].copy()
        
        if len(df) < 252:  # Need at least 1 year of data
            return pd.DataFrame()
        
        features_list = []
        
        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol].sort_index()
            
            if len(sym_df) < 200:  # Need enough history
                continue
            
            # Get latest data point
            latest = sym_df.iloc[-1]
            latest_date = sym_df.index[-1]
            
            # Price data
            close = sym_df['close'].values
            high = sym_df['high'].values
            low = sym_df['low'].values
            volume = sym_df['volume'].values
            
            # Basic returns
            returns = pd.Series(close).pct_change()
            
            features = {
                'symbol': symbol,
                'date': latest_date,
                'close': latest['close'],
                
                # Tier 1: Momentum
                'return_1d': returns.iloc[-1] if len(returns) > 1 else 0,
                'return_5d': (close[-1] / close[-5] - 1) if len(close) > 5 else 0,
                'return_10d': (close[-1] / close[-10] - 1) if len(close) > 10 else 0,
                'return_20d': (close[-1] / close[-20] - 1) if len(close) > 20 else 0,
                
                # Moving averages
                'sma_5': np.mean(close[-5:]) if len(close) >= 5 else close[-1],
                'sma_20': np.mean(close[-20:]) if len(close) >= 20 else close[-1],
                'sma_50': np.mean(close[-50:]) if len(close) >= 50 else close[-1],
                'sma_200': np.mean(close[-200:]) if len(close) >= 200 else close[-1],
                
                # Volatility
                'volatility_20d': returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0,
                
                # RSI
                'rsi_14': self._calculate_rsi(close, 14),
                
                # Volume
                'volume_ratio': volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 and np.mean(volume[-20:]) > 0 else 1,
                
                # Tier 2: Additional
                'price_vs_sma200': close[-1] / np.mean(close[-200:]) - 1 if len(close) >= 200 else 0,
                'week52_high': max(high[-252:]) if len(high) >= 252 else high[-1],
                'week52_high_ratio': close[-1] / max(high[-252:]) if len(high) >= 252 else 1,
            }
            
            # Derived ratios
            features['sma_5_ratio'] = close[-1] / features['sma_5'] if features['sma_5'] > 0 else 1
            features['sma_20_ratio'] = close[-1] / features['sma_20'] if features['sma_20'] > 0 else 1
            
            # Market context features (if available)
            if self.spy_data is not None and len(self.spy_data) > 0:
                spy_filtered = self.spy_data[self.spy_data.index <= as_of_date]
                if len(spy_filtered) >= 200:
                    spy_close = spy_filtered['close'].values
                    spy_sma200 = np.mean(spy_close[-200:])
                    features['market_regime'] = 1 if spy_close[-1] > spy_sma200 else 0
                    
                    # Correlation to SPY
                    if len(returns) >= 20:
                        spy_returns = spy_filtered['close'].pct_change().iloc[-20:].values
                        stock_returns = returns.iloc[-20:].values
                        if len(spy_returns) == len(stock_returns):
                            try:
                                corr = np.corrcoef(stock_returns, spy_returns)[0, 1]
                                features['corr_to_spy'] = corr if not np.isnan(corr) else 0
                            except:
                                features['corr_to_spy'] = 0
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# ============================================================================
# WALK-FORWARD BACKTESTER
# ============================================================================

class WalkForwardBacktester:
    """
    Walk-forward backtesting with proper out-of-sample validation.
    
    Walk-Forward Process:
    1. Train on [0, T1], test on [T1, T2]
    2. Train on [0, T2], test on [T2, T3]  (expanding window)
    3. ... continue until end of data
    
    This ensures:
    - Model only sees past data during training
    - Performance is measured on truly out-of-sample data
    - Multiple test periods provide statistical confidence
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        train_window_days: int = 252,  # 1 year training
        test_window_days: int = 21,    # ~1 month testing
        retrain_frequency: int = 21,   # Retrain monthly
        max_position_pct: float = 0.10,
        top_k: int = 5,
        slippage_bps: float = 10.0,    # 10 bps slippage (conservative)
        commission_per_trade: float = 1.0,  # $1 per trade
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
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
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # Model and features
        self.model = None
        self.feature_calculator = FeatureCalculator()
        self.feature_columns: List[str] = []
        
        # Data storage
        self.price_data: Optional[pd.DataFrame] = None
        self.spy_data: Optional[pd.DataFrame] = None
        
    def _fetch_data(self) -> None:
        """Fetch all required historical data."""
        logger.info(f"Fetching data for {len(self.tickers)} tickers...")
        
        # Calculate start date with buffer for training
        data_start = self.start_date - timedelta(days=self.train_window_days + 100)
        
        all_data = []
        
        # Fetch ticker data
        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker, 
                    start=data_start.strftime('%Y-%m-%d'),
                    end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    progress=False
                )
                if not df.empty:
                    df = df.reset_index()
                    df['symbol'] = ticker
                    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                    df = df.rename(columns={'adj close': 'adj_close'})
                    df = df.set_index('date')
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
        
        if not all_data:
            raise ValueError("No data fetched for any ticker")
        
        self.price_data = pd.concat(all_data)
        logger.info(f"Fetched {len(self.price_data)} rows of data")
        
        # Fetch SPY for benchmark
        try:
            spy = yf.download(
                'SPY',
                start=data_start.strftime('%Y-%m-%d'),
                end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
            spy = spy.reset_index()
            spy.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in spy.columns]
            spy = spy.set_index('date')
            self.spy_data = spy
        except Exception as e:
            logger.warning(f"Failed to fetch SPY: {e}")
            self.spy_data = pd.DataFrame()
        
        # Fetch VIX for context
        try:
            vix = yf.download(
                '^VIX',
                start=data_start.strftime('%Y-%m-%d'),
                end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
            vix = vix.reset_index()
            vix.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in vix.columns]
            vix = vix.set_index('date')
            self.feature_calculator.set_market_data(self.spy_data, vix)
        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")
    
    def _train_model(self, as_of_date: datetime) -> None:
        """Train model using only data available up to as_of_date."""
        logger.info(f"Training model as of {as_of_date.strftime('%Y-%m-%d')}...")
        
        # Get features for training period
        train_start = as_of_date - timedelta(days=self.train_window_days)
        
        # Calculate features for each day in training period
        # (This is computationally expensive but necessary for proper validation)
        features_list = []
        
        current = train_start
        while current <= as_of_date:
            if current.weekday() < 5:  # Skip weekends
                day_features = self.feature_calculator.calculate_features(
                    self.price_data, current
                )
                if not day_features.empty:
                    features_list.append(day_features)
            current += timedelta(days=1)
        
        if not features_list:
            logger.warning("No features available for training")
            return
        
        features_df = pd.concat(features_list, ignore_index=True)
        
        # Prepare training data
        # Target: will price be higher in 1 day?
        features_df = features_df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Create target (next day return positive)
        features_df['target'] = features_df.groupby('symbol')['close'].shift(-1) > features_df['close']
        features_df = features_df.dropna(subset=['target'])
        features_df['target'] = features_df['target'].astype(int)
        
        # Feature columns (exclude metadata)
        exclude_cols = ['symbol', 'date', 'close', 'target', 'week52_high']
        self.feature_columns = [c for c in features_df.columns if c not in exclude_cols]
        
        X = features_df[self.feature_columns].fillna(0)
        y = features_df['target']
        
        if len(X) < 100:
            logger.warning(f"Insufficient training data: {len(X)} rows")
            return
        
        # Train LightGBM
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X, y)
        logger.info(f"Model trained on {len(X)} samples")
    
    def _get_signals(self, as_of_date: datetime) -> Dict[str, float]:
        """Generate signals for current date."""
        if self.model is None:
            return {}
        
        # Get features for current date
        features = self.feature_calculator.calculate_features(self.price_data, as_of_date)
        
        if features.empty:
            return {}
        
        # Ensure we have all required columns
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        X = features[self.feature_columns].fillna(0)
        
        # Get probabilities
        try:
            probas = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return {}
        
        signals = {}
        for i, row in features.iterrows():
            signals[row['symbol']] = probas[i]
        
        return signals
    
    def _get_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get closing price for symbol on date."""
        try:
            sym_data = self.price_data[self.price_data['symbol'] == symbol]
            date_data = sym_data[sym_data.index == date]
            if not date_data.empty:
                return float(date_data['close'].iloc[0])
            
            # Try nearby dates if exact date not found
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
    
    def _execute_buy(self, symbol: str, price: float, date: datetime) -> bool:
        """Execute a buy order."""
        if symbol in self.positions:
            return False
        
        exec_price = self._apply_slippage(price, 'buy')
        max_value = self.cash * self.max_position_pct
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
            entry_date=date
        )
        
        logger.debug(f"BUY {shares} {symbol} @ ${exec_price:.2f}")
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
            holding_days=holding_days
        ))
        
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
                positions_value += pos.cost_basis  # Use cost basis if price unavailable
        
        return self.cash + positions_value
    
    def run(self) -> WalkForwardResult:
        """Run the walk-forward backtest."""
        logger.info(f"Starting walk-forward backtest: {self.start_date} to {self.end_date}")
        
        # Fetch all data
        self._fetch_data()
        
        # Initialize
        current_date = self.start_date
        last_train_date = None
        n_train_periods = 0
        trading_days = []
        
        # Main simulation loop
        while current_date <= self.end_date:
            # Skip weekends
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
            
            # Get signals
            signals = self._get_signals(current_date)
            
            if signals:
                # Sort by signal strength
                sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
                
                # Close positions below threshold or not in top K
                top_symbols = [s[0] for s in sorted_signals[:self.top_k]]
                
                for symbol in list(self.positions.keys()):
                    signal = signals.get(symbol, 0.5)
                    price = self._get_price(symbol, current_date)
                    
                    if price and (signal < self.sell_threshold or symbol not in top_symbols):
                        self._execute_sell(symbol, price, current_date)
                
                # Open new positions for strong signals
                for symbol, prob in sorted_signals[:self.top_k]:
                    if prob >= self.buy_threshold and symbol not in self.positions:
                        price = self._get_price(symbol, current_date)
                        if price:
                            self._execute_buy(symbol, price, current_date)
            
            # Record equity
            portfolio_value = self._get_portfolio_value(current_date)
            self.equity_curve.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': self.cash,
                'n_positions': len(self.positions)
            })
            
            current_date += timedelta(days=1)
        
        # Close all positions at end
        final_date = trading_days[-1] if trading_days else self.end_date
        for symbol in list(self.positions.keys()):
            price = self._get_price(symbol, final_date)
            if price:
                self._execute_sell(symbol, price, final_date)
        
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
            spy_start = self.spy_data[self.spy_data.index >= self.start_date]['close'].iloc[0]
            spy_end = self.spy_data[self.spy_data.index <= self.end_date]['close'].iloc[-1]
            benchmark_return = spy_end - spy_start
            benchmark_return_pct = (spy_end / spy_start) - 1
            
            # Benchmark equity curve
            benchmark_df = self.spy_data[
                (self.spy_data.index >= self.start_date) & 
                (self.spy_data.index <= self.end_date)
            ].copy()
            benchmark_df['value'] = self.initial_capital * (benchmark_df['close'] / spy_start)
        else:
            benchmark_return = 0
            benchmark_return_pct = 0
            benchmark_df = pd.DataFrame()
        
        alpha = annualized_return - (benchmark_return_pct / years if years > 0 else 0)
        
        # Risk metrics
        daily_returns = equity_df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (daily_returns.mean() * 252) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        equity_df['peak'] = equity_df['value'].cummax()
        equity_df['drawdown'] = equity_df['value'] - equity_df['peak']
        equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak']
        
        max_dd = abs(equity_df['drawdown'].min())
        max_dd_pct = abs(equity_df['drawdown_pct'].min())
        
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
        
        # Statistical significance vs benchmark
        if len(daily_returns) > 30 and len(benchmark_df) > 0:
            benchmark_returns = benchmark_df['close'].pct_change().dropna()
            # Align dates
            common_dates = daily_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 30:
                strat_rets = daily_returns.loc[common_dates]
                bench_rets = benchmark_returns.loc[common_dates]
                excess_returns = strat_rets - bench_rets.values
                t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
            else:
                t_stat, p_value = 0, 1
        else:
            t_stat, p_value = 0, 1
        
        metrics = BacktestMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            benchmark_return_pct=benchmark_return_pct,
            alpha=alpha,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
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
            t_statistic=t_stat,
            p_value=p_value
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
            test_window_days=self.test_window_days
        )


# ============================================================================
# HTML REPORT GENERATOR
# ============================================================================

def generate_backtest_report(result: WalkForwardResult, output_path: str = None) -> str:
    """Generate an HTML report for backtest results."""
    
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 
            'backtest_report.html'
        )
    
    m = result.metrics
    
    # Prepare equity curve data for chart
    equity_dates = result.equity_curve.index.strftime('%Y-%m-%d').tolist()
    equity_values = result.equity_curve['value'].tolist()
    
    # Benchmark data
    if not result.benchmark_curve.empty:
        bench_dates = result.benchmark_curve.index.strftime('%Y-%m-%d').tolist()
        bench_values = result.benchmark_curve['value'].tolist()
    else:
        bench_dates = equity_dates
        bench_values = [result.initial_capital] * len(equity_dates)
    
    # Drawdown data
    dd_values = (result.drawdown_curve['drawdown_pct'] * 100).tolist()
    
    # Trade list
    trades_html = ''
    for t in result.trades[-50:]:  # Last 50 trades
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
        </tr>
        '''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Walk-Forward Backtest Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #22c55e, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ text-align: center; color: #71717a; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .card h2 {{ font-size: 0.85rem; color: #71717a; margin-bottom: 12px; text-transform: uppercase; }}
        .metric {{ font-size: 2rem; font-weight: 700; }}
        .metric.positive {{ color: #22c55e; }}
        .metric.negative {{ color: #ef4444; }}
        .metric-label {{ color: #71717a; font-size: 0.8rem; margin-top: 4px; }}
        .full-width {{ grid-column: 1 / -1; }}
        .chart-container {{ height: 300px; position: relative; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        th {{ color: #71717a; font-weight: 500; text-transform: uppercase; font-size: 0.7rem; }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .warning {{ background: rgba(245, 158, 11, 0.2); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #f59e0b; }}
        .success {{ background: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #22c55e; }}
        .stats-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .stat {{ flex: 1; min-width: 100px; text-align: center; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 600; }}
        .stat-label {{ font-size: 0.7rem; color: #71717a; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Walk-Forward Backtest Report</h1>
        <p class="subtitle">
            {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')} | 
            {result.n_train_periods} training periods | 
            Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </p>
        
        {'<div class="warning"><strong>WARNING:</strong> Strategy did NOT outperform benchmark (SPY). P-value: ' + f'{m.p_value:.4f}' + ' - results are not statistically significant.</div>' if m.alpha < 0 or m.p_value > 0.05 else '<div class="success"><strong>POSITIVE ALPHA:</strong> Strategy outperformed benchmark by ' + f'{m.alpha:.2%}' + ' (annualized). P-value: ' + f'{m.p_value:.4f}' + '</div>'}
        
        <div class="grid">
            <div class="card">
                <h2>Total Return</h2>
                <div class="metric {'positive' if m.total_return >= 0 else 'negative'}">{m.total_return_pct:+.2%}</div>
                <div class="metric-label">${m.total_return:+,.2f}</div>
            </div>
            <div class="card">
                <h2>Benchmark (SPY)</h2>
                <div class="metric {'positive' if m.benchmark_return_pct >= 0 else 'negative'}">{m.benchmark_return_pct:+.2%}</div>
                <div class="metric-label">Buy & Hold</div>
            </div>
            <div class="card">
                <h2>Alpha</h2>
                <div class="metric {'positive' if m.alpha >= 0 else 'negative'}">{m.alpha:+.2%}</div>
                <div class="metric-label">Excess Return (ann.)</div>
            </div>
            <div class="card">
                <h2>Sharpe Ratio</h2>
                <div class="metric {'positive' if m.sharpe_ratio >= 1 else 'negative' if m.sharpe_ratio < 0 else ''}">{m.sharpe_ratio:.2f}</div>
                <div class="metric-label">Risk-Adjusted</div>
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
        
        <div class="grid">
            <div class="card">
                <h2>Risk Metrics</h2>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-value">{m.sortino_ratio:.2f}</div>
                        <div class="stat-label">Sortino</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value negative">{m.max_drawdown_pct:.1%}</div>
                        <div class="stat-label">Max DD</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{m.volatility:.1%}</div>
                        <div class="stat-label">Volatility</div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>Trade Statistics</h2>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-value">{m.total_trades}</div>
                        <div class="stat-label">Total Trades</div>
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
            </div>
        </div>
        
        <div class="grid">
            <div class="card full-width">
                <h2>Recent Trades</h2>
                <div style="overflow-x: auto;">
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
                    pointRadius: 0
                }}, {{
                    label: 'SPY (Benchmark)',
                    data: {json.dumps(bench_values)},
                    borderColor: '#71717a',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#e4e4e7' }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#71717a', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y: {{ ticks: {{ color: '#71717a', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
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
                    backgroundColor: 'rgba(239, 68, 68, 0.3)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#71717a', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y: {{ ticks: {{ color: '#71717a', callback: v => v.toFixed(1) + '%' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Backtest report saved to {output_path}")
    return output_path


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_walkforward_backtest(
    tickers: List[str],
    start_date: str,
    end_date: str,
    **kwargs
) -> WalkForwardResult:
    """
    Run a walk-forward backtest.
    
    Args:
        tickers: List of ticker symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        **kwargs: Additional arguments for WalkForwardBacktester
        
    Returns:
        WalkForwardResult with comprehensive metrics
    """
    backtester = WalkForwardBacktester(tickers, start_date, end_date, **kwargs)
    result = backtester.run()
    
    # Generate HTML report
    generate_backtest_report(result)
    
    return result


if __name__ == '__main__':
    # Example: Run a 1-year backtest
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
    
    result = run_walkforward_backtest(
        tickers=tickers,
        start_date='2024-01-01',
        end_date='2024-12-01',
        initial_capital=100000,
    )
    
    print(result.summary())
