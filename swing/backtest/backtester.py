"""
Backtesting engine for strategy validation.

Simulates historical trading with realistic assumptions:
- Transaction costs
- Slippage
- Gap handling
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from swing.config import get_settings
from swing.data.stock_feed import StockFeed
from swing.signals.options_flow_strategy import Signal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a backtested trade."""

    symbol: str
    direction: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    exit_reason: str
    signal_confidence: float


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    # P&L metrics
    total_pnl: float
    total_return_pct: float
    avg_win: float
    avg_loss: float

    # Risk metrics
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_duration: float

    # Trade list
    trades: List[BacktestTrade] = field(default_factory=list)

    # Equity curve
    equity_curve: List[Dict] = field(default_factory=list)


class Backtester:
    """
    Backtesting engine for options flow strategy.

    Per spec (Backtesting Realism):
    - Include slippage: 0.1-0.5% per trade based on liquidity
    - Account for gaps: If stock gaps past stop, use worst-case price
    - No commissions for Alpaca
    """

    # Realistic slippage assumption
    SLIPPAGE_PCT = 0.002  # 0.2%

    def __init__(
        self,
        initial_capital: Optional[float] = None,
        risk_per_trade: Optional[float] = None,
        max_positions: Optional[int] = None,
        stop_loss_pct: Optional[float] = None,
        profit_target_pct: Optional[float] = None,
    ):
        settings = get_settings()

        self.initial_capital = initial_capital or settings.initial_capital
        self.risk_per_trade = risk_per_trade or settings.risk_per_trade
        self.max_positions = max_positions or settings.max_positions
        self.stop_loss_pct = stop_loss_pct or settings.default_stop_loss_pct
        self.profit_target_pct = profit_target_pct or settings.profit_target_pct

        self.stock_feed = StockFeed()

    def _apply_slippage(self, price: float, direction: str, is_entry: bool) -> float:
        """
        Apply slippage to price.

        Entry: Pay more (buy) or receive less (short)
        Exit: Receive less (sell) or pay more (cover)
        """
        if is_entry:
            if direction == "LONG":
                return price * (1 + self.SLIPPAGE_PCT)
            else:  # SHORT
                return price * (1 - self.SLIPPAGE_PCT)
        else:  # Exit
            if direction == "LONG":
                return price * (1 - self.SLIPPAGE_PCT)
            else:  # SHORT
                return price * (1 + self.SLIPPAGE_PCT)

    def _calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
    ) -> int:
        """Calculate shares based on risk."""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        risk_amount = capital * self.risk_per_trade
        shares_by_risk = int(risk_amount / risk_per_share)

        max_position = capital * 0.20
        shares_by_position = int(max_position / entry_price)

        return min(shares_by_risk, shares_by_position)

    def run_backtest(
        self,
        signals: List[Signal],
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        """
        Run backtest on historical data with given signals.

        Args:
            signals: List of historical signals with timestamps
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with full statistics
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")

        capital = self.initial_capital
        trades: List[BacktestTrade] = []
        equity_curve: List[Dict] = []
        open_positions: Dict[str, Dict] = {}

        # Get unique symbols
        symbols = list(set(s.symbol for s in signals))

        # Fetch historical data for all symbols
        price_data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = self.stock_feed.get_historical(
                    symbol, start_date=start_date, end_date=end_date
                )
                price_data[symbol] = df
            except Exception as e:
                logger.warning(f"Could not fetch data for {symbol}: {e}")

        # Sort signals by timestamp
        signals = sorted(signals, key=lambda s: s.timestamp)

        # Process each day
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            # Get today's signals
            today_signals = [
                s for s in signals
                if s.timestamp.date() == current_date.date()
                and s.is_actionable()
            ]

            # Update open positions and check for exits
            positions_to_close = []
            for symbol, pos in open_positions.items():
                if symbol not in price_data:
                    continue

                try:
                    # Get today's price
                    df = price_data[symbol]
                    if current_date not in df.index:
                        # Find nearest date
                        closest = df.index[df.index <= current_date]
                        if len(closest) == 0:
                            continue
                        today_idx = closest[-1]
                    else:
                        today_idx = current_date

                    today_data = df.loc[today_idx]
                    current_price = float(today_data["Close"])
                    today_low = float(today_data["Low"])
                    today_high = float(today_data["High"])

                    pos["days_held"] += 1

                    # Check stop loss (use low for long, high for short)
                    if pos["direction"] == "LONG":
                        if today_low <= pos["stop_loss"]:
                            # Gap handling: use worst case
                            exit_price = min(today_low, pos["stop_loss"])
                            exit_price = self._apply_slippage(
                                exit_price, pos["direction"], is_entry=False
                            )
                            positions_to_close.append(
                                (symbol, exit_price, "STOP_HIT")
                            )
                            continue

                        if today_high >= pos["profit_target"]:
                            exit_price = self._apply_slippage(
                                pos["profit_target"], pos["direction"], is_entry=False
                            )
                            positions_to_close.append(
                                (symbol, exit_price, "PROFIT_TARGET")
                            )
                            continue
                    else:  # SHORT
                        if today_high >= pos["stop_loss"]:
                            exit_price = max(today_high, pos["stop_loss"])
                            exit_price = self._apply_slippage(
                                exit_price, pos["direction"], is_entry=False
                            )
                            positions_to_close.append(
                                (symbol, exit_price, "STOP_HIT")
                            )
                            continue

                        if today_low <= pos["profit_target"]:
                            exit_price = self._apply_slippage(
                                pos["profit_target"], pos["direction"], is_entry=False
                            )
                            positions_to_close.append(
                                (symbol, exit_price, "PROFIT_TARGET")
                            )
                            continue

                    # Check expiry (5 days)
                    if pos["days_held"] >= 5:
                        exit_price = self._apply_slippage(
                            current_price, pos["direction"], is_entry=False
                        )
                        positions_to_close.append((symbol, exit_price, "EXPIRY"))

                except Exception as e:
                    logger.debug(f"Error checking position {symbol}: {e}")

            # Close positions
            for symbol, exit_price, reason in positions_to_close:
                pos = open_positions.pop(symbol)

                if pos["direction"] == "LONG":
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                else:
                    pnl = (pos["entry_price"] - exit_price) * pos["shares"]

                return_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                if pos["direction"] == "SHORT":
                    return_pct = -return_pct

                trade = BacktestTrade(
                    symbol=symbol,
                    direction=pos["direction"],
                    entry_date=pos["entry_date"],
                    exit_date=current_date,
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    shares=pos["shares"],
                    pnl=pnl,
                    return_pct=return_pct,
                    exit_reason=reason,
                    signal_confidence=pos["confidence"],
                )
                trades.append(trade)
                capital += pnl

            # Open new positions from today's signals
            for signal in today_signals:
                if len(open_positions) >= self.max_positions:
                    break
                if signal.symbol in open_positions:
                    continue
                if signal.symbol not in price_data:
                    continue

                try:
                    df = price_data[signal.symbol]
                    if current_date not in df.index:
                        closest = df.index[df.index <= current_date]
                        if len(closest) == 0:
                            continue
                        today_idx = closest[-1]
                    else:
                        today_idx = current_date

                    today_data = df.loc[today_idx]
                    entry_price = float(today_data["Close"])

                    # Apply slippage
                    entry_price = self._apply_slippage(
                        entry_price, signal.direction.value, is_entry=True
                    )

                    # Calculate stops
                    if signal.direction == SignalDirection.LONG:
                        stop_loss = entry_price * (1 - self.stop_loss_pct)
                        profit_target = entry_price * (1 + self.profit_target_pct)
                    else:
                        stop_loss = entry_price * (1 + self.stop_loss_pct)
                        profit_target = entry_price * (1 - self.profit_target_pct)

                    # Calculate position size
                    shares = self._calculate_position_size(
                        capital, entry_price, stop_loss
                    )
                    if shares <= 0:
                        continue

                    open_positions[signal.symbol] = {
                        "direction": signal.direction.value,
                        "entry_price": entry_price,
                        "shares": shares,
                        "stop_loss": stop_loss,
                        "profit_target": profit_target,
                        "entry_date": current_date,
                        "days_held": 0,
                        "confidence": signal.confidence,
                    }

                except Exception as e:
                    logger.debug(f"Error opening position {signal.symbol}: {e}")

            # Record equity
            equity_curve.append({
                "date": current_date.isoformat(),
                "capital": capital,
                "open_positions": len(open_positions),
            })

            current_date += timedelta(days=1)

        # Calculate metrics
        result = self._calculate_metrics(
            trades=trades,
            equity_curve=equity_curve,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=capital,
        )

        return result

    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[Dict],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        final_capital: float,
    ) -> BacktestResult:
        """Calculate backtest metrics."""
        if not trades:
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                total_pnl=0.0,
                total_return_pct=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                avg_trade_duration=0.0,
                trades=trades,
                equity_curve=equity_curve,
            )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = len(wins) / len(trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(abs(t.pnl) for t in losses) / len(losses) if losses else 0.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0

        total_pnl = sum(t.pnl for t in trades)
        total_return_pct = (final_capital - initial_capital) / initial_capital

        # Max drawdown
        peak = initial_capital
        max_dd = 0.0
        for point in equity_curve:
            if point["capital"] > peak:
                peak = point["capital"]
            dd = (peak - point["capital"]) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                prev = equity_curve[i - 1]["capital"]
                curr = equity_curve[i]["capital"]
                returns.append((curr - prev) / prev)

            if returns:
                from statistics import mean, stdev

                avg_ret = mean(returns)
                std_ret = stdev(returns) if len(returns) > 1 else 1.0
                sharpe = (avg_ret / std_ret) * (252 ** 0.5) if std_ret > 0 else 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Average duration
        avg_duration = sum(
            (t.exit_date - t.entry_date).days for t in trades
        ) / len(trades)

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2),
            total_pnl=round(total_pnl, 2),
            total_return_pct=round(total_return_pct * 100, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            avg_trade_duration=round(avg_duration, 1),
            trades=trades,
            equity_curve=equity_curve,
        )

    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results summary."""
        print(f"""
Backtest Results
================
Period: {result.start_date.date()} to {result.end_date.date()}
Initial Capital: ${result.initial_capital:,.2f}
Final Capital: ${result.final_capital:,.2f}

Trade Statistics
----------------
Total Trades: {result.total_trades}
Winning: {result.winning_trades} | Losing: {result.losing_trades}
Win Rate: {result.win_rate:.1%}
Profit Factor: {result.profit_factor:.2f}x

P&L
---
Total P&L: ${result.total_pnl:,.2f}
Total Return: {result.total_return_pct:.1f}%
Avg Win: ${result.avg_win:,.2f}
Avg Loss: ${result.avg_loss:,.2f}

Risk Metrics
------------
Max Drawdown: {result.max_drawdown_pct:.1f}%
Sharpe Ratio: {result.sharpe_ratio:.2f}
Avg Trade Duration: {result.avg_trade_duration:.1f} days

Target Comparison (from spec):
- Win Rate: {result.win_rate:.1%} (target: 50-55%)
- Profit Factor: {result.profit_factor:.2f}x (target: 1.5-2.0x)
- Sharpe: {result.sharpe_ratio:.2f} (target: 0.8-1.2)
""")
