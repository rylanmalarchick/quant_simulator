"""
Tests for options flow strategy signal generation.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from swing.data.options_feed import OptionsMetrics
from swing.signals.options_flow_strategy import (
    OptionsFlowStrategy,
    Signal,
    SignalDirection,
)


@pytest.fixture
def strategy():
    """Create a strategy instance with mocked settings."""
    with patch("swing.signals.options_flow_strategy.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            min_confidence=0.60,
            signal_expiry_days=5,
        )
        yield OptionsFlowStrategy()


@pytest.fixture
def bullish_metrics():
    """Create metrics that should trigger a bullish signal."""
    return OptionsMetrics(
        symbol="AAPL",
        timestamp=datetime.now(),
        underlying_price=150.0,
        put_call_ratio=0.3,  # < 0.5 (bullish)
        put_call_ratio_zscore=-1.5,
        gamma_exposure=5_000_000,  # > 1M (bullish)
        iv_skew=-0.1,
        put_volume=1000,
        call_volume=3333,
        put_volume_spike=0.5,  # < 1.0 (bullish)
        call_volume_spike=2.0,  # > 1.5 (bullish)
        unusual_activity=True,
        flow_score=0.75,
        components={},
    )


@pytest.fixture
def bearish_metrics():
    """Create metrics that should trigger a bearish signal."""
    return OptionsMetrics(
        symbol="TSLA",
        timestamp=datetime.now(),
        underlying_price=200.0,
        put_call_ratio=2.0,  # > 1.5 (bearish)
        put_call_ratio_zscore=2.0,
        gamma_exposure=-5_000_000,  # < -1M (bearish)
        iv_skew=0.25,  # > 0.20 (bearish)
        put_volume=5000,
        call_volume=2500,
        put_volume_spike=2.5,  # > 2.0 (bearish)
        call_volume_spike=0.5,
        unusual_activity=True,
        flow_score=0.80,
        components={},
    )


@pytest.fixture
def neutral_metrics():
    """Create metrics that should not trigger any signal."""
    return OptionsMetrics(
        symbol="MSFT",
        timestamp=datetime.now(),
        underlying_price=300.0,
        put_call_ratio=0.8,  # Between thresholds
        put_call_ratio_zscore=0.0,
        gamma_exposure=500_000,  # Below threshold
        iv_skew=0.05,
        put_volume=2000,
        call_volume=2500,
        put_volume_spike=1.0,
        call_volume_spike=1.0,
        unusual_activity=False,
        flow_score=0.3,
        components={},
    )


class TestOptionsFlowStrategy:
    """Test signal generation logic."""

    def test_bullish_signal_conditions(self, strategy, bullish_metrics):
        """Test that bullish conditions are correctly detected."""
        assert strategy._check_bullish_conditions(bullish_metrics) is True
        assert strategy._check_bearish_conditions(bullish_metrics) is False

    def test_bearish_signal_conditions(self, strategy, bearish_metrics):
        """Test that bearish conditions are correctly detected."""
        assert strategy._check_bearish_conditions(bearish_metrics) is True
        assert strategy._check_bullish_conditions(bearish_metrics) is False

    def test_neutral_signal_conditions(self, strategy, neutral_metrics):
        """Test that neutral metrics don't trigger signals."""
        assert strategy._check_bullish_conditions(neutral_metrics) is False
        assert strategy._check_bearish_conditions(neutral_metrics) is False

    def test_generate_bullish_signal(self, strategy, bullish_metrics):
        """Test generating a bullish signal."""
        signal = strategy.generate_signal(bullish_metrics)

        assert signal.symbol == "AAPL"
        assert signal.direction == SignalDirection.LONG
        assert signal.confidence > 0
        assert signal.expiry_days == 5
        assert "pcr_signal" in signal.components
        assert "gamma_signal" in signal.components

    def test_generate_bearish_signal(self, strategy, bearish_metrics):
        """Test generating a bearish signal."""
        signal = strategy.generate_signal(bearish_metrics)

        assert signal.symbol == "TSLA"
        assert signal.direction == SignalDirection.SHORT
        assert signal.confidence > 0

    def test_generate_neutral_signal(self, strategy, neutral_metrics):
        """Test generating a neutral (no) signal."""
        signal = strategy.generate_signal(neutral_metrics)

        assert signal.direction == SignalDirection.NONE
        assert signal.confidence == 0.0

    def test_signal_actionable(self, strategy, bullish_metrics):
        """Test actionable signal detection."""
        signal = strategy.generate_signal(bullish_metrics)

        # Should be actionable if confidence >= threshold
        assert signal.direction != SignalDirection.NONE

    def test_generate_signals_multiple(self, strategy, bullish_metrics, bearish_metrics, neutral_metrics):
        """Test generating signals for multiple symbols."""
        metrics_list = [bullish_metrics, bearish_metrics, neutral_metrics]
        signals = strategy.generate_signals(metrics_list)

        assert len(signals) == 3
        assert signals[0].direction == SignalDirection.LONG
        assert signals[1].direction == SignalDirection.SHORT
        assert signals[2].direction == SignalDirection.NONE

    def test_get_actionable_signals(self, strategy, bullish_metrics, neutral_metrics):
        """Test filtering to only actionable signals."""
        metrics_list = [bullish_metrics, neutral_metrics]
        actionable = strategy.get_actionable_signals(metrics_list, min_confidence=0.01)

        # Only the bullish signal should be actionable
        assert len(actionable) == 1
        assert actionable[0].symbol == "AAPL"

    def test_pcr_normalization_bullish(self, strategy):
        """Test PCR signal normalization for bullish direction."""
        # Very low PCR should give high score
        score = strategy._normalize_pcr_signal(0.1, -2.0, SignalDirection.LONG)
        assert score > 0.7

        # PCR at threshold should give low score
        score = strategy._normalize_pcr_signal(0.5, 0.0, SignalDirection.LONG)
        assert score == 0.0

    def test_gamma_normalization(self, strategy):
        """Test gamma signal normalization."""
        # High positive gamma should give high bullish score
        score = strategy._normalize_gamma_signal(10_000_000, SignalDirection.LONG)
        assert score == 1.0

        # Below threshold should give zero
        score = strategy._normalize_gamma_signal(500_000, SignalDirection.LONG)
        assert score == 0.0

    def test_rank_signals(self, strategy):
        """Test signal ranking by confidence."""
        signal1 = Signal(
            symbol="A",
            direction=SignalDirection.LONG,
            confidence=0.5,
            expiry_days=5,
            entry_level="market",
            timestamp=datetime.now(),
        )
        signal2 = Signal(
            symbol="B",
            direction=SignalDirection.LONG,
            confidence=0.8,
            expiry_days=5,
            entry_level="market",
            timestamp=datetime.now(),
        )
        signal3 = Signal(
            symbol="C",
            direction=SignalDirection.SHORT,
            confidence=0.65,
            expiry_days=5,
            entry_level="market",
            timestamp=datetime.now(),
        )

        ranked = strategy.rank_signals([signal1, signal2, signal3])

        assert ranked[0].symbol == "B"
        assert ranked[1].symbol == "C"
        assert ranked[2].symbol == "A"
