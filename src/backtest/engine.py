"""
Backtesting Engine for quantSim

This module provides a proper event-driven backtesting framework that:
1. Simulates trading day-by-day without look-ahead bias
2. Tracks portfolio value, positions, and cash
3. Calculates real performance metrics (P&L, Sharpe, drawdown, etc.)
4. Uses walk-forward training to prevent data leakage
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from src.ingest.market_data_client import get_market_data
from src.ingest.sqlite_writer import get_db_connection, write_raw_bars
from src.features.builder import build_and_persist_features
from src.config import load_config, load_tickers
from src.logging import get_logger
import pickle
import os
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    side: str  # 'long' or 'short'
    
    @property
    def value(self) -> float:
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


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    equity_curve: pd.DataFrame
    trades: List[Trade]
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        return f"""
========== Backtest Results ==========
Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
Initial Capital: ${self.initial_capital:,.2f}
Final Value: ${self.final_value:,.2f}

--- Returns ---
Total Return: ${self.total_return:,.2f} ({self.total_return_pct:.2%})
Annualized Return: {self.annualized_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2%})

--- Trades ---
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.2%}
Winning Trades: {self.winning_trades}
Losing Trades: {self.losing_trades}
Avg Win: ${self.avg_win:,.2f}
Avg Loss: ${self.avg_loss:,.2f}
Profit Factor: {self.profit_factor:.2f}
==========================================
"""


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Simulates trading by:
    1. Walking through each trading day
    2. Training model on historical data (walk-forward)
    3. Generating signals based on current features
    4. Executing trades with realistic constraints
    5. Tracking portfolio performance
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.10,
        top_k: int = 5,
        retrain_frequency: int = 30,  # Retrain every N days
        slippage_bps: float = 5.0,  # 5 basis points slippage
        commission_per_share: float = 0.01,
    ):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.top_k = top_k
        self.retrain_frequency = retrain_frequency
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # Model state
        self.model = None
        self.last_train_date: Optional[datetime] = None
        
        # Load config
        self.config = load_config()
        self.tickers = load_tickers()
        
    def _get_all_tickers(self) -> List[str]:
        """Get all tickers from config."""
        all_tickers = []
        for category in ['funds', 'quantum']:
            if category in self.tickers:
                all_tickers.extend(self.tickers[category])
        return list(set(all_tickers))
    
    def _fetch_historical_data(self, lookback_days: int = 365) -> pd.DataFrame:
        """Fetch historical data for all tickers."""
        tickers = self._get_all_tickers()
        all_data = []
        
        end_date = self.start_date
        start_date = end_date - timedelta(days=lookback_days)
        
        for ticker in tickers:
            data = get_market_data(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            if data and data.get('c'):
                for i in range(len(data['t'])):
                    all_data.append({
                        'symbol': ticker,
                        'timestamp': data['t'][i],
                        'open': data['o'][i],
                        'high': data['h'][i],
                        'low': data['l'][i],
                        'close': data['c'][i],
                        'volume': data['v'][i]
                    })
        
        return pd.DataFrame(all_data)
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for the given price data."""
        features_list = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('timestamp').copy()
            
            if len(symbol_df) < 20:
                continue
                
            # Calculate technical features
            close = symbol_df['close'].values
            high = symbol_df['high'].values
            low = symbol_df['low'].values
            volume = symbol_df['volume'].values
            
            # Returns
            returns_1d = pd.Series(close).pct_change(1).values
            returns_5d = pd.Series(close).pct_change(5).values
            returns_20d = pd.Series(close).pct_change(20).values
            
            # Moving averages
            sma_5 = pd.Series(close).rolling(5).mean().values
            sma_20 = pd.Series(close).rolling(20).mean().values
            
            # Volatility
            volatility_20d = pd.Series(returns_1d).rolling(20).std().values
            
            # RSI
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).values
            
            # Volume features
            volume_sma = pd.Series(volume).rolling(20).mean().values
            volume_ratio = volume / np.where(volume_sma > 0, volume_sma, 1)
            
            for i in range(len(symbol_df)):
                if i < 20:  # Skip rows without enough history
                    continue
                    
                features_list.append({
                    'symbol': symbol,
                    'timestamp': symbol_df.iloc[i]['timestamp'],
                    'close': close[i],
                    'return_1d': returns_1d[i] if not np.isnan(returns_1d[i]) else 0,
                    'return_5d': returns_5d[i] if not np.isnan(returns_5d[i]) else 0,
                    'return_20d': returns_20d[i] if not np.isnan(returns_20d[i]) else 0,
                    'sma_5_ratio': close[i] / sma_5[i] if sma_5[i] > 0 else 1,
                    'sma_20_ratio': close[i] / sma_20[i] if sma_20[i] > 0 else 1,
                    'volatility_20d': volatility_20d[i] if not np.isnan(volatility_20d[i]) else 0,
                    'rsi': rsi[i] if not np.isnan(rsi[i]) else 50,
                    'volume_ratio': volume_ratio[i] if not np.isnan(volume_ratio[i]) else 1,
                })
        
        return pd.DataFrame(features_list)
    
    def _train_model(self, features_df: pd.DataFrame) -> None:
        """Train or retrain the model using walk-forward validation."""
        logger.info("Training model for backtest...")
        
        # Sort by timestamp
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
        
        # Prepare features and target
        feature_cols = [c for c in features_df.columns if c not in ['symbol', 'timestamp', 'close']]
        X = features_df[feature_cols].fillna(0)
        y = (features_df['close'].shift(-1) > features_df['close']).astype(int)
        
        # Remove last row
        X = X.iloc[:-1]
        y = y.iloc[:-1]
        
        # Train model
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X, y)
        logger.info("Model trained successfully")
    
    def _generate_signals(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals for current day."""
        if self.model is None:
            return {}
        
        # Get latest features for each symbol
        latest = features_df.groupby('symbol').last().reset_index()
        
        feature_cols = [c for c in latest.columns if c not in ['symbol', 'timestamp', 'close']]
        X = latest[feature_cols].fillna(0)
        
        # Predict probabilities
        probas = self.model.predict_proba(X)[:, 1]
        
        signals = {}
        for i, row in latest.iterrows():
            signals[row['symbol']] = probas[i]
        
        return signals
    
    def _get_current_price(self, symbol: str, date: datetime, prices: Dict[str, float]) -> Optional[float]:
        """Get current price for a symbol."""
        return prices.get(symbol)
    
    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price."""
        slippage_mult = self.slippage_bps / 10000
        if side == 'buy':
            return price * (1 + slippage_mult)
        else:
            return price * (1 - slippage_mult)
    
    def _calculate_commission(self, shares: int) -> float:
        """Calculate commission for a trade."""
        return shares * self.commission_per_share
    
    def _execute_buy(self, symbol: str, price: float, date: datetime) -> bool:
        """Execute a buy order."""
        if symbol in self.positions:
            return False
        
        # Calculate position size
        max_position_value = self.cash * self.max_position_pct
        exec_price = self._apply_slippage(price, 'buy')
        shares = int(max_position_value / exec_price)
        
        if shares <= 0:
            return False
        
        # Calculate costs
        cost = shares * exec_price
        commission = self._calculate_commission(shares)
        total_cost = cost + commission
        
        if total_cost > self.cash:
            shares = int((self.cash - commission) / exec_price)
            if shares <= 0:
                return False
            cost = shares * exec_price
            commission = self._calculate_commission(shares)
            total_cost = cost + commission
        
        # Execute
        self.cash -= total_cost
        self.positions[symbol] = Position(
            symbol=symbol,
            shares=shares,
            entry_price=exec_price,
            entry_date=date,
            side='long'
        )
        
        logger.debug(f"BUY {shares} {symbol} @ ${exec_price:.2f}")
        return True
    
    def _execute_sell(self, symbol: str, price: float, date: datetime) -> bool:
        """Execute a sell order (close position)."""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        exec_price = self._apply_slippage(price, 'sell')
        
        # Calculate proceeds
        proceeds = position.shares * exec_price
        commission = self._calculate_commission(position.shares)
        net_proceeds = proceeds - commission
        
        # Calculate P&L
        pnl = net_proceeds - (position.shares * position.entry_price)
        pnl_pct = pnl / (position.shares * position.entry_price)
        
        # Record trade
        self.trades.append(Trade(
            symbol=symbol,
            side='long',
            shares=position.shares,
            entry_price=position.entry_price,
            exit_price=exec_price,
            entry_date=position.entry_date,
            exit_date=date,
            pnl=pnl,
            pnl_percent=pnl_pct
        ))
        
        # Update cash
        self.cash += net_proceeds
        del self.positions[symbol]
        
        logger.debug(f"SELL {position.shares} {symbol} @ ${exec_price:.2f}, P&L: ${pnl:.2f}")
        return True
    
    def _calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            pos.shares * prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def run(self) -> BacktestResult:
        """Run the backtest."""
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Fetch historical data for initial training
        logger.info("Fetching historical data...")
        historical_df = self._fetch_historical_data(lookback_days=365)
        
        if historical_df.empty:
            raise ValueError("No historical data available for backtesting")
        
        # Calculate initial features and train model
        features_df = self._calculate_features(historical_df)
        self._train_model(features_df)
        self.last_train_date = self.start_date
        
        # Walk through each trading day
        current_date = self.start_date
        tickers = self._get_all_tickers()
        days_since_train = 0
        
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            # Fetch current day's prices
            prices = {}
            for ticker in tickers:
                data = get_market_data(
                    ticker,
                    current_date.strftime('%Y-%m-%d'),
                    (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
                )
                if data and data.get('c') and len(data['c']) > 0:
                    prices[ticker] = data['c'][0]
            
            if not prices:
                current_date += timedelta(days=1)
                continue
            
            # Retrain model periodically
            days_since_train += 1
            if days_since_train >= self.retrain_frequency:
                logger.info(f"Retraining model at {current_date}")
                # Fetch updated historical data
                lookback_start = current_date - timedelta(days=365)
                new_data = []
                for ticker in tickers:
                    data = get_market_data(
                        ticker,
                        lookback_start.strftime('%Y-%m-%d'),
                        current_date.strftime('%Y-%m-%d')
                    )
                    if data and data.get('c'):
                        for i in range(len(data['t'])):
                            new_data.append({
                                'symbol': ticker,
                                'timestamp': data['t'][i],
                                'open': data['o'][i],
                                'high': data['h'][i],
                                'low': data['l'][i],
                                'close': data['c'][i],
                                'volume': data['v'][i]
                            })
                if new_data:
                    new_df = pd.DataFrame(new_data)
                    features_df = self._calculate_features(new_df)
                    self._train_model(features_df)
                    days_since_train = 0
            
            # Generate signals
            signals = self._generate_signals(features_df)
            
            # Sort by signal strength
            sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            
            # Close positions that are no longer in top signals or have poor signals
            current_holdings = list(self.positions.keys())
            top_symbols = [s[0] for s in sorted_signals[:self.top_k]]
            
            for symbol in current_holdings:
                if symbol not in top_symbols or signals.get(symbol, 0) < 0.4:
                    if symbol in prices:
                        self._execute_sell(symbol, prices[symbol], current_date)
            
            # Open new positions for top signals
            for symbol, prob in sorted_signals[:self.top_k]:
                if prob > 0.6 and symbol not in self.positions:
                    if symbol in prices:
                        self._execute_buy(symbol, prices[symbol], current_date)
            
            # Record equity
            portfolio_value = self._calculate_portfolio_value(prices)
            self.equity_curve.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })
            
            current_date += timedelta(days=1)
        
        # Close all remaining positions at end
        if self.equity_curve:
            last_prices = {pos.symbol: pos.entry_price for pos in self.positions.values()}
            for symbol in list(self.positions.keys()):
                self._execute_sell(symbol, last_prices.get(symbol, 0), self.end_date)
        
        return self._calculate_results()
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results and metrics."""
        equity_df = pd.DataFrame(self.equity_curve)
        
        if equity_df.empty:
            return BacktestResult(
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                final_value=self.initial_capital,
                total_return=0,
                total_return_pct=0,
                annualized_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                win_rate=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                equity_curve=equity_df,
                trades=self.trades
            )
        
        final_value = equity_df['value'].iloc[-1]
        total_return = final_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Annualized return
        days = (self.end_date - self.start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return_pct) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio (assuming 252 trading days, risk-free rate of 0)
        equity_df['returns'] = equity_df['value'].pct_change()
        daily_returns = equity_df['returns'].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        equity_df['peak'] = equity_df['value'].cummax()
        equity_df['drawdown'] = equity_df['value'] - equity_df['peak']
        equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak']
        max_drawdown = abs(equity_df['drawdown'].min())
        max_drawdown_pct = abs(equity_df['drawdown_pct'].min())
        
        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in self.trades if t.pnl <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        total_wins = sum(wins)
        total_losses = sum(losses)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_df,
            trades=self.trades
        )


def run_backtest(start_date_str: str, end_date_str: str, **kwargs) -> BacktestResult:
    """
    Convenience function to run a backtest.
    
    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        **kwargs: Additional arguments passed to BacktestEngine
        
    Returns:
        BacktestResult with performance metrics
    """
    engine = BacktestEngine(start_date_str, end_date_str, **kwargs)
    return engine.run()


if __name__ == '__main__':
    # Example: Run a 6-month backtest
    result = run_backtest('2024-01-01', '2024-06-30')
    print(result.summary())
