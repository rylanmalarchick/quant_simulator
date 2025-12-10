"""
Tests for position manager and risk controls.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from swing.risk.position_manager import PositionManager, Position, ExitReason
from swing.signals.options_flow_strategy import Signal, SignalDirection


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.initial_capital = 25000.0
    settings.risk_per_trade = 0.02
    settings.max_positions = 5
    settings.max_daily_loss = 0.05
    settings.max_position_pct = 0.20
    settings.default_stop_loss_pct = 0.02
    settings.profit_target_pct = 0.10
    return settings


@pytest.fixture
def position_manager(mock_settings):
    """Create a position manager with mocked settings."""
    with patch("swing.risk.position_manager.get_settings") as mock_get:
        mock_get.return_value = mock_settings
        yield PositionManager()


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing."""
    return Signal(
        symbol="AAPL",
        direction=SignalDirection.LONG,
        confidence=0.75,
        expiry_days=5,
        entry_level="market",
        timestamp=datetime.now(),
        components={"pcr_signal": 0.8, "gamma_signal": 0.7},
    )


class TestPositionManager:
    """Test position management logic."""

    def test_initialization(self, position_manager):
        """Test position manager initialization."""
        assert position_manager.total_capital == 25000.0
        assert position_manager.risk_per_trade == 0.02
        assert position_manager.max_positions == 5
        assert len(position_manager.positions) == 0

    def test_risk_amount(self, position_manager):
        """Test risk amount calculation."""
        # 2% of $25,000 = $500
        assert position_manager.risk_amount == 500.0

    def test_max_position_value(self, position_manager):
        """Test max position value calculation."""
        # 20% of $25,000 = $5,000
        assert position_manager.max_position_value == 5000.0

    def test_can_take_trade_success(self, position_manager):
        """Test that we can take a trade when conditions are met."""
        can_trade, reason = position_manager.can_take_trade("AAPL")
        assert can_trade is True
        assert reason == "OK"

    def test_can_take_trade_already_in_position(self, position_manager, sample_signal):
        """Test that we can't open duplicate positions."""
        # Open a position first
        position_manager.open_position(sample_signal, entry_price=150.0)

        can_trade, reason = position_manager.can_take_trade("AAPL")
        assert can_trade is False
        assert "Already have position" in reason

    def test_can_take_trade_max_positions(self, position_manager, mock_settings):
        """Test that we can't exceed max positions."""
        with patch("swing.risk.position_manager.get_settings") as mock_get:
            mock_get.return_value = mock_settings

            # Fill up positions
            for i in range(5):
                signal = Signal(
                    symbol=f"SYM{i}",
                    direction=SignalDirection.LONG,
                    confidence=0.75,
                    expiry_days=5,
                    entry_level="market",
                    timestamp=datetime.now(),
                )
                position_manager.open_position(signal, entry_price=100.0 + i)

        can_trade, reason = position_manager.can_take_trade("NEWSTOCK")
        assert can_trade is False
        assert "Max positions" in reason

    def test_calculate_stop_loss_long(self, position_manager):
        """Test stop loss calculation for long positions."""
        stop = position_manager.calculate_stop_loss(100.0, SignalDirection.LONG)
        # 2% below entry
        assert stop == 98.0

    def test_calculate_stop_loss_short(self, position_manager):
        """Test stop loss calculation for short positions."""
        stop = position_manager.calculate_stop_loss(100.0, SignalDirection.SHORT)
        # 2% above entry
        assert stop == 102.0

    def test_calculate_profit_target_long(self, position_manager):
        """Test profit target calculation for long positions."""
        target = position_manager.calculate_profit_target(100.0, SignalDirection.LONG)
        # 10% above entry
        assert abs(target - 110.0) < 0.001

    def test_calculate_position_size(self, position_manager):
        """Test position size calculation."""
        # Entry $150, Stop $147 (2% stop)
        # Risk per share: $3
        # Risk amount: $500
        # Shares by risk: 500/3 = 166
        # Max by position: $5,000 / $150 = 33
        # Should be capped at 33
        shares = position_manager.calculate_position_size(150.0, 147.0)
        assert shares == 33

    def test_calculate_position_size_risk_limited(self, position_manager):
        """Test position size when risk is the limiting factor."""
        # Entry $10, Stop $9.50 (5% stop)
        # Risk per share: $0.50
        # Risk amount: $500
        # Shares by risk: 500/0.50 = 1000
        # Max by position: $5,000 / $10 = 500
        # Should be capped at 500
        shares = position_manager.calculate_position_size(10.0, 9.50)
        assert shares == 500

    def test_open_position(self, position_manager, sample_signal):
        """Test opening a position."""
        position = position_manager.open_position(sample_signal, entry_price=150.0)

        assert position is not None
        assert position.symbol == "AAPL"
        assert position.direction == SignalDirection.LONG
        assert position.entry_price == 150.0
        assert position.shares > 0
        assert position.stop_loss == 147.0
        assert position.profit_target == 165.0

        assert "AAPL" in position_manager.positions

    def test_should_exit_stop_loss_long(self, position_manager, sample_signal):
        """Test stop loss exit for long position."""
        position_manager.open_position(sample_signal, entry_price=150.0)

        # Price drops below stop
        exit_reason = position_manager.should_exit("AAPL", 146.0)
        assert exit_reason == ExitReason.STOP_HIT

    def test_should_exit_profit_target_long(self, position_manager, sample_signal):
        """Test profit target exit for long position."""
        position_manager.open_position(sample_signal, entry_price=150.0)

        # Price rises above target
        exit_reason = position_manager.should_exit("AAPL", 166.0)
        assert exit_reason == ExitReason.PROFIT_TARGET

    def test_should_exit_no_exit(self, position_manager, sample_signal):
        """Test no exit when price is in range."""
        position_manager.open_position(sample_signal, entry_price=150.0)

        # Price within range
        exit_reason = position_manager.should_exit("AAPL", 155.0)
        assert exit_reason is None

    def test_should_exit_expiry(self, position_manager, sample_signal):
        """Test expiry exit."""
        position_manager.open_position(sample_signal, entry_price=150.0)

        # Simulate days passing
        position_manager.positions["AAPL"].days_held = 5

        exit_reason = position_manager.should_exit("AAPL", 155.0)
        assert exit_reason == ExitReason.EXPIRY

    def test_close_position(self, position_manager, sample_signal):
        """Test closing a position."""
        position_manager.open_position(sample_signal, entry_price=150.0)

        trade = position_manager.close_position("AAPL", 160.0, ExitReason.PROFIT_TARGET)

        assert trade is not None
        assert trade["symbol"] == "AAPL"
        assert trade["direction"] == "LONG"
        assert trade["entry_price"] == 150.0
        assert trade["exit_price"] == 160.0
        assert trade["realized_pnl"] > 0
        assert trade["win"] == 1

        assert "AAPL" not in position_manager.positions

    def test_portfolio_summary(self, position_manager, sample_signal):
        """Test portfolio summary."""
        position_manager.open_position(sample_signal, entry_price=150.0)
        position_manager.positions["AAPL"].update_price(155.0)

        summary = position_manager.get_portfolio_summary()

        assert summary["positions_count"] == 1
        assert summary["total_capital"] == 25000.0
        assert summary["total_unrealized_pnl"] > 0
        assert "AAPL" in summary["positions"]


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            entry_price=150.0,
            shares=100,
            stop_loss=147.0,
            profit_target=165.0,
            entry_time=datetime.now(),
            expiry_days=5,
        )

        assert pos.current_price == 150.0
        assert pos.unrealized_pnl == 0.0
        assert pos.days_held == 0

    def test_position_update_price(self):
        """Test updating position price."""
        pos = Position(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            entry_price=150.0,
            shares=100,
            stop_loss=147.0,
            profit_target=165.0,
            entry_time=datetime.now(),
            expiry_days=5,
        )

        pos.update_price(160.0)

        assert pos.current_price == 160.0
        assert pos.unrealized_pnl == 1000.0  # (160 - 150) * 100
        assert pos.highest_price == 160.0

    def test_position_pnl_percent(self):
        """Test P&L percentage calculation."""
        pos = Position(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            shares=100,
            stop_loss=98.0,
            profit_target=110.0,
            entry_time=datetime.now(),
            expiry_days=5,
        )

        pos.update_price(105.0)
        assert pos.pnl_percent == 0.05  # 5%

    def test_position_value(self):
        """Test position value calculation."""
        pos = Position(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            shares=50,
            stop_loss=98.0,
            profit_target=110.0,
            entry_time=datetime.now(),
            expiry_days=5,
        )

        pos.update_price(110.0)
        assert pos.position_value == 5500.0
        assert pos.entry_value == 5000.0
