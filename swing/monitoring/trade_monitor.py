"""
Trade monitoring and P&L tracking.

Tracks:
- Real-time unrealized P&L
- Realized P&L per trade
- Win rate, profit factor, and other metrics
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev
from typing import Dict, List, Optional

from swing.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a closed trade."""

    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: int
    realized_pnl: float
    return_pct: float
    entry_time: datetime
    exit_time: datetime
    duration_days: int
    exit_reason: str
    win: bool


@dataclass
class PositionSnapshot:
    """Snapshot of open position."""

    symbol: str
    direction: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_percent: float


class TradeMonitor:
    """
    Monitors trades and calculates performance metrics.

    Per spec:
    - Target win rate: 50-55%
    - Target profit factor: 1.5-2.0x
    - Target Sharpe: 0.8-1.2
    """

    def __init__(self, initial_capital: Optional[float] = None):
        settings = get_settings()
        self.initial_capital = initial_capital or settings.initial_capital

        # Trade history
        self.trades: List[TradeRecord] = []

        # Current state
        self.open_positions: Dict[str, PositionSnapshot] = {}
        self.current_equity: float = self.initial_capital

        # Daily tracking
        self.daily_pnl: float = 0.0
        self.realized_pnl_today: float = 0.0
        self.daily_pnl_history: List[float] = []

        # Monthly tracking
        self.monthly_pnl: float = 0.0
        self.monthly_pnl_history: List[float] = []

    @property
    def total_realized_pnl(self) -> float:
        """Sum of all realized P&L."""
        return sum(t.realized_pnl for t in self.trades)

    @property
    def total_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L in open positions."""
        return sum(pos.unrealized_pnl for pos in self.open_positions.values())

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage of initial capital."""
        total_pnl = self.total_realized_pnl + self.total_unrealized_pnl
        return total_pnl / self.initial_capital

    def update_position(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        entry_price: float,
        current_price: float,
    ) -> None:
        """Update or add a position snapshot."""
        if direction.upper() == "LONG":
            unrealized_pnl = (current_price - entry_price) * quantity
        else:
            unrealized_pnl = (entry_price - current_price) * quantity

        pnl_percent = (current_price - entry_price) / entry_price
        if direction.upper() == "SHORT":
            pnl_percent = -pnl_percent

        self.open_positions[symbol] = PositionSnapshot(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            pnl_percent=pnl_percent,
        )

    def remove_position(self, symbol: str) -> None:
        """Remove a position from tracking."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]

    def record_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        entry_time: datetime,
        exit_time: datetime,
        exit_reason: str,
    ) -> TradeRecord:
        """Record a closed trade."""
        if direction.upper() == "LONG":
            realized_pnl = (exit_price - entry_price) * quantity
        else:
            realized_pnl = (entry_price - exit_price) * quantity

        return_pct = (exit_price - entry_price) / entry_price
        if direction.upper() == "SHORT":
            return_pct = -return_pct

        duration_days = (exit_time - entry_time).days

        trade = TradeRecord(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            realized_pnl=realized_pnl,
            return_pct=return_pct,
            entry_time=entry_time,
            exit_time=exit_time,
            duration_days=duration_days,
            exit_reason=exit_reason,
            win=realized_pnl > 0,
        )

        self.trades.append(trade)
        self.realized_pnl_today += realized_pnl

        # Remove from open positions
        self.remove_position(symbol)

        logger.info(
            f"Trade recorded: {symbol} {direction} P&L=${realized_pnl:.2f} ({return_pct:.1%})"
        )

        return trade

    def update_realtime(self, equity: float) -> None:
        """
        Update real-time tracking.

        Called periodically during market hours.
        """
        self.current_equity = equity

        # Calculate daily P&L
        self.daily_pnl = self.realized_pnl_today + self.total_unrealized_pnl

        # Check daily loss limit
        settings = get_settings()
        daily_loss_limit = self.initial_capital * settings.max_daily_loss

        if self.daily_pnl < -daily_loss_limit:
            logger.warning(
                f"DAILY LOSS LIMIT: P&L=${self.daily_pnl:.2f} exceeds "
                f"-${daily_loss_limit:.2f}"
            )

    def end_of_day(self) -> None:
        """Called at end of trading day."""
        self.daily_pnl_history.append(self.daily_pnl)
        self.monthly_pnl += self.daily_pnl

        logger.info(f"End of day P&L: ${self.daily_pnl:.2f}")

        # Reset daily tracking
        self.realized_pnl_today = 0.0
        self.daily_pnl = 0.0

    def end_of_month(self) -> None:
        """Called at end of month."""
        self.monthly_pnl_history.append(self.monthly_pnl)

        logger.info(f"End of month P&L: ${self.monthly_pnl:.2f}")

        self.monthly_pnl = 0.0

    def get_metrics(self) -> Dict:
        """
        Calculate performance metrics.

        Returns dict with:
        - total_trades, win_rate, profit_factor
        - avg_win, avg_loss
        - total_pnl, total_return_pct
        - sharpe_ratio (if enough data)
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
            }

        # Win/loss counts
        wins = [t for t in self.trades if t.win]
        losses = [t for t in self.trades if not t.win]

        win_rate = len(wins) / len(self.trades) if self.trades else 0.0

        # Average win/loss
        avg_win = mean([t.realized_pnl for t in wins]) if wins else 0.0
        avg_loss = mean([abs(t.realized_pnl) for t in losses]) if losses else 0.0

        # Profit factor
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive(True)
        consecutive_losses = self._calculate_consecutive(False)

        # Sharpe ratio (if we have enough daily returns)
        sharpe_ratio = self._calculate_sharpe() if len(self.daily_pnl_history) >= 5 else 0.0

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "total_pnl": round(self.total_realized_pnl, 2),
            "total_return_pct": round(self.total_return_pct * 100, 2),
            "current_equity": round(self.current_equity, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "monthly_pnl": round(self.monthly_pnl, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
            "open_positions": len(self.open_positions),
        }

    def _calculate_consecutive(self, win: bool) -> int:
        """Calculate max consecutive wins or losses."""
        if not self.trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in self.trades:
            if trade.win == win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if len(self.daily_pnl_history) < 5:
            return 0.0

        # Convert P&L to returns
        returns = [pnl / self.initial_capital for pnl in self.daily_pnl_history]

        avg_return = mean(returns)
        std_return = stdev(returns) if len(returns) > 1 else 1.0

        if std_return == 0:
            return 0.0

        # Annualize (252 trading days)
        sharpe = (avg_return / std_return) * (252 ** 0.5)

        return sharpe

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown as percentage."""
        if not self.daily_pnl_history:
            return 0.0

        # Calculate cumulative equity
        equity_curve = [self.initial_capital]
        for pnl in self.daily_pnl_history:
            equity_curve.append(equity_curve[-1] + pnl)

        # Find max drawdown
        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd

    def get_summary(self) -> str:
        """Get human-readable summary."""
        metrics = self.get_metrics()

        return f"""
Trading Performance Summary
===========================
Total Trades: {metrics['total_trades']} ({metrics['wins']}W / {metrics['losses']}L)
Win Rate: {metrics['win_rate']:.1%}
Profit Factor: {metrics['profit_factor']:.2f}x
Avg Win: ${metrics['avg_win']:.2f}
Avg Loss: ${metrics['avg_loss']:.2f}

P&L
---
Total P&L: ${metrics['total_pnl']:.2f}
Total Return: {metrics['total_return_pct']:.1f}%
Daily P&L: ${metrics['daily_pnl']:.2f}
Monthly P&L: ${metrics['monthly_pnl']:.2f}

Risk Metrics
------------
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown_pct']:.1f}%
Max Consecutive Wins: {metrics['consecutive_wins']}
Max Consecutive Losses: {metrics['consecutive_losses']}

Current State
-------------
Equity: ${metrics['current_equity']:.2f}
Open Positions: {metrics['open_positions']}
"""
