"""
Position Manager for risk management.

Handles:
- Position sizing based on risk per trade
- Stop loss and profit target management
- Daily loss limits
- Max position constraints
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from swing.config import get_settings
from swing.signals.options_flow_strategy import Signal, SignalDirection

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reason for exiting a position."""

    STOP_HIT = "STOP_HIT"
    PROFIT_TARGET = "PROFIT_TARGET"
    EXPIRY = "EXPIRY"
    MANUAL = "MANUAL"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"


@dataclass
class Position:
    """Active position tracking."""

    symbol: str
    direction: SignalDirection
    entry_price: float
    shares: int
    stop_loss: float
    profit_target: float
    entry_time: datetime
    expiry_days: int

    # Tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    days_held: int = 0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = 0.0  # For short positions

    # Metadata
    signal_confidence: float = 0.0
    signal_components: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.current_price = self.entry_price
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price

    @property
    def position_value(self) -> float:
        """Current position value."""
        return self.current_price * self.shares

    @property
    def entry_value(self) -> float:
        """Original position value at entry."""
        return self.entry_price * self.shares

    @property
    def pnl_percent(self) -> float:
        """P&L as percentage of entry."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == SignalDirection.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - self.current_price) / self.entry_price

    def update_price(self, price: float) -> None:
        """Update current price and tracking."""
        self.current_price = price

        if self.direction == SignalDirection.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.shares
            self.highest_price = max(self.highest_price, price)
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - price) * self.shares
            self.lowest_price = min(self.lowest_price, price)


class PositionManager:
    """
    Manages positions with risk controls.

    Per spec:
    - Risk per Trade: 2% = $500 loss max per trade (on $25K)
    - Max Positions: 5 simultaneous
    - Max Daily Loss: 5% = $1,250 daily limit
    - Max Position Size: 20% of capital = $5,000
    """

    def __init__(
        self,
        total_capital: Optional[float] = None,
        risk_per_trade: Optional[float] = None,
        max_positions: Optional[int] = None,
        max_daily_loss: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        default_stop_loss_pct: Optional[float] = None,
        profit_target_pct: Optional[float] = None,
    ):
        settings = get_settings()

        self.total_capital = total_capital or settings.initial_capital
        self.risk_per_trade = risk_per_trade or settings.risk_per_trade
        self.max_positions = max_positions or settings.max_positions
        self.max_daily_loss = max_daily_loss or settings.max_daily_loss
        self.max_position_pct = max_position_pct or settings.max_position_pct
        self.default_stop_loss_pct = default_stop_loss_pct or settings.default_stop_loss_pct
        self.profit_target_pct = profit_target_pct or settings.profit_target_pct

        # Active positions
        self.positions: Dict[str, Position] = {}

        # Daily tracking
        self.daily_pnl: float = 0.0
        self.realized_pnl_today: float = 0.0
        self.daily_pnl_history: List[float] = []

        # Closed trades
        self.closed_trades: List[Dict] = []

    @property
    def risk_amount(self) -> float:
        """Dollar amount to risk per trade."""
        return self.total_capital * self.risk_per_trade

    @property
    def max_position_value(self) -> float:
        """Maximum dollar value for a single position."""
        return self.total_capital * self.max_position_pct

    @property
    def daily_loss_limit(self) -> float:
        """Maximum daily loss in dollars."""
        return self.total_capital * self.max_daily_loss

    @property
    def total_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def total_position_value(self) -> float:
        """Total value of all open positions."""
        return sum(pos.position_value for pos in self.positions.values())

    def can_take_trade(self, symbol: str) -> tuple[bool, str]:
        """
        Check if we can take a new trade.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Already in this position
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"

        # Max positions reached
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) reached"

        # Daily loss limit hit
        total_daily = self.realized_pnl_today + self.total_unrealized_pnl
        if total_daily < -self.daily_loss_limit:
            return False, f"Daily loss limit (${self.daily_loss_limit:.0f}) exceeded"

        return True, "OK"

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: SignalDirection,
        stop_loss_pct: Optional[float] = None,
    ) -> float:
        """Calculate stop loss price."""
        pct = stop_loss_pct or self.default_stop_loss_pct

        if direction == SignalDirection.LONG:
            return entry_price * (1 - pct)
        else:  # SHORT
            return entry_price * (1 + pct)

    def calculate_profit_target(
        self,
        entry_price: float,
        direction: SignalDirection,
        target_pct: Optional[float] = None,
    ) -> float:
        """Calculate profit target price."""
        pct = target_pct or self.profit_target_pct

        if direction == SignalDirection.LONG:
            return entry_price * (1 + pct)
        else:  # SHORT
            return entry_price * (1 - pct)

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
    ) -> int:
        """
        Calculate number of shares to buy.

        Uses Kelly-inspired approach:
        - Risk per share = distance to stop
        - Shares = risk_amount / risk_per_share
        - Capped at max_position_pct of capital
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0

        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        # Shares to risk target amount
        shares_by_risk = int(self.risk_amount / risk_per_share)

        # Max shares by position size limit
        shares_by_position = int(self.max_position_value / entry_price)

        # Take the smaller of the two
        shares = min(shares_by_risk, shares_by_position)

        logger.debug(
            f"Position sizing: entry=${entry_price:.2f}, stop=${stop_loss:.2f}, "
            f"risk_per_share=${risk_per_share:.2f}, shares={shares}"
        )

        return max(shares, 0)

    def open_position(
        self,
        signal: Signal,
        entry_price: float,
        stop_loss_pct: Optional[float] = None,
        profit_target_pct: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Open a new position based on signal.

        Returns Position if opened, None if rejected.
        """
        can_trade, reason = self.can_take_trade(signal.symbol)
        if not can_trade:
            logger.warning(f"Cannot open position for {signal.symbol}: {reason}")
            return None

        # Calculate stops and targets
        stop_loss = self.calculate_stop_loss(
            entry_price, signal.direction, stop_loss_pct
        )
        profit_target = self.calculate_profit_target(
            entry_price, signal.direction, profit_target_pct
        )

        # Calculate position size
        shares = self.calculate_position_size(entry_price, stop_loss)
        if shares <= 0:
            logger.warning(f"Position size is 0 for {signal.symbol}")
            return None

        position = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            profit_target=profit_target,
            entry_time=datetime.now(),
            expiry_days=signal.expiry_days,
            signal_confidence=signal.confidence,
            signal_components=signal.components,
        )

        self.positions[signal.symbol] = position

        logger.info(
            f"Opened {signal.direction.value} position: {shares} shares of {signal.symbol} "
            f"@ ${entry_price:.2f}, stop=${stop_loss:.2f}, target=${profit_target:.2f}"
        )

        return position

    def should_exit(self, symbol: str, current_price: float) -> Optional[ExitReason]:
        """
        Check if position should be exited.

        Returns ExitReason if should exit, None otherwise.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pos.update_price(current_price)

        # Check stop loss
        if pos.direction == SignalDirection.LONG:
            if current_price <= pos.stop_loss:
                return ExitReason.STOP_HIT
            if current_price >= pos.profit_target:
                return ExitReason.PROFIT_TARGET
        else:  # SHORT
            if current_price >= pos.stop_loss:
                return ExitReason.STOP_HIT
            if current_price <= pos.profit_target:
                return ExitReason.PROFIT_TARGET

        # Check expiry
        if pos.days_held >= pos.expiry_days:
            return ExitReason.EXPIRY

        # Check daily loss limit
        total_daily = self.realized_pnl_today + self.total_unrealized_pnl
        if total_daily < -self.daily_loss_limit:
            return ExitReason.DAILY_LOSS_LIMIT

        return None

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: ExitReason,
    ) -> Optional[Dict]:
        """
        Close a position and record the trade.

        Returns trade record dict.
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        pos = self.positions[symbol]
        pos.update_price(exit_price)

        # Calculate realized P&L
        if pos.direction == SignalDirection.LONG:
            realized_pnl = (exit_price - pos.entry_price) * pos.shares
        else:  # SHORT
            realized_pnl = (pos.entry_price - exit_price) * pos.shares

        # Record trade
        trade = {
            "symbol": symbol,
            "direction": pos.direction.value,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "shares": pos.shares,
            "realized_pnl": realized_pnl,
            "return_pct": pos.pnl_percent,
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "days_held": pos.days_held,
            "exit_reason": reason.value,
            "signal_confidence": pos.signal_confidence,
            "win": 1 if realized_pnl > 0 else 0,
        }

        self.closed_trades.append(trade)
        self.realized_pnl_today += realized_pnl

        # Remove position
        del self.positions[symbol]

        logger.info(
            f"Closed {trade['direction']} position: {symbol} "
            f"@ ${exit_price:.2f}, P&L: ${realized_pnl:.2f} ({trade['return_pct']:.1%}), "
            f"Reason: {reason.value}"
        )

        return trade

    def update_day(self) -> None:
        """
        Called at end of day to update tracking.

        - Increment days_held for all positions
        - Record daily P&L
        - Reset daily realized P&L
        """
        for pos in self.positions.values():
            pos.days_held += 1

        total_pnl = self.realized_pnl_today + self.total_unrealized_pnl
        self.daily_pnl_history.append(total_pnl)

        self.daily_pnl = total_pnl
        self.realized_pnl_today = 0.0

        logger.info(f"End of day: P&L=${total_pnl:.2f}, positions={len(self.positions)}")

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio state."""
        return {
            "total_capital": self.total_capital,
            "positions_count": len(self.positions),
            "max_positions": self.max_positions,
            "total_position_value": self.total_position_value,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "realized_pnl_today": self.realized_pnl_today,
            "daily_pnl": self.realized_pnl_today + self.total_unrealized_pnl,
            "daily_loss_limit": self.daily_loss_limit,
            "closed_trades_count": len(self.closed_trades),
            "positions": {
                symbol: {
                    "direction": pos.direction.value,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "pnl_percent": pos.pnl_percent,
                    "days_held": pos.days_held,
                }
                for symbol, pos in self.positions.items()
            },
        }
