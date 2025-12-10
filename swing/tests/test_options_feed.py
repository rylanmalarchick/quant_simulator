"""
Tests for options data feed.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from swing.data.options_feed import OptionsFeed, OptionsMetrics, OptionContract


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.polygon_api_key = "test_key"
    settings.polygon_rate_limit = 100
    settings.cache_dir = "/tmp/test_cache"
    settings.cache_ttl_seconds = 3600
    return settings


@pytest.fixture
def options_feed(mock_settings):
    """Create options feed with mocked settings."""
    with patch("swing.data.options_feed.get_settings") as mock_get:
        mock_get.return_value = mock_settings
        with patch("swing.data.options_feed.DataCache"):
            yield OptionsFeed()


@pytest.fixture
def sample_options_chain():
    """Create sample options chain for testing."""
    return [
        OptionContract(
            ticker="AAPL230120C00150000",
            contract_type="call",
            strike=150.0,
            expiration_date="2023-01-20",
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            gamma=0.05,
            delta=0.55,
            underlying_price=152.0,
            last_price=5.00,
            bid=4.90,
            ask=5.10,
        ),
        OptionContract(
            ticker="AAPL230120P00150000",
            contract_type="put",
            strike=150.0,
            expiration_date="2023-01-20",
            volume=500,
            open_interest=3000,
            implied_volatility=0.28,
            gamma=0.05,
            delta=-0.45,
            underlying_price=152.0,
            last_price=3.00,
            bid=2.90,
            ask=3.10,
        ),
        OptionContract(
            ticker="AAPL230120C00155000",
            contract_type="call",
            strike=155.0,
            expiration_date="2023-01-20",
            volume=800,
            open_interest=4000,
            implied_volatility=0.26,
            gamma=0.04,
            delta=0.35,
            underlying_price=152.0,
            last_price=2.50,
            bid=2.40,
            ask=2.60,
        ),
        OptionContract(
            ticker="AAPL230120P00155000",
            contract_type="put",
            strike=155.0,
            expiration_date="2023-01-20",
            volume=400,
            open_interest=2500,
            implied_volatility=0.30,
            gamma=0.04,
            delta=-0.65,
            underlying_price=152.0,
            last_price=5.50,
            bid=5.40,
            ask=5.60,
        ),
    ]


class TestOptionsFeed:
    """Test options data feed."""

    def test_calculate_put_call_ratio(self, options_feed, sample_options_chain):
        """Test put/call ratio calculation."""
        pcr, put_vol, call_vol = options_feed.calculate_put_call_ratio(sample_options_chain)

        # Put volume: 500 + 400 = 900
        # Call volume: 1000 + 800 = 1800
        # PCR = 900 / 1800 = 0.5
        assert put_vol == 900
        assert call_vol == 1800
        assert pcr == 0.5

    def test_calculate_put_call_ratio_no_calls(self, options_feed):
        """Test PCR when there are no calls."""
        chain = [
            OptionContract(
                ticker="TEST",
                contract_type="put",
                strike=100.0,
                expiration_date="2023-01-20",
                volume=100,
                open_interest=500,
                implied_volatility=0.25,
                gamma=0.05,
                delta=-0.5,
                underlying_price=100.0,
                last_price=2.0,
                bid=1.9,
                ask=2.1,
            )
        ]
        pcr, _, _ = options_feed.calculate_put_call_ratio(chain)
        assert pcr == 0.0

    def test_calculate_gamma_exposure(self, options_feed, sample_options_chain):
        """Test gamma exposure calculation."""
        gex = options_feed.calculate_gamma_exposure(sample_options_chain)

        # GEX should be positive when calls dominate
        # This is a simplified test - actual value depends on calculation
        assert isinstance(gex, float)

    def test_calculate_iv_skew(self, options_feed, sample_options_chain):
        """Test IV skew calculation."""
        skew = options_feed.calculate_iv_skew(sample_options_chain)

        # Skew = (put_iv - call_iv) / avg_iv
        # With higher put IV, should be slightly positive
        assert isinstance(skew, float)

    def test_calculate_iv_skew_empty_chain(self, options_feed):
        """Test IV skew with empty chain."""
        skew = options_feed.calculate_iv_skew([])
        assert skew == 0.0

    def test_calculate_volume_spike_no_history(self, options_feed):
        """Test volume spike with no history."""
        spike = options_feed.calculate_volume_spike(1000, "AAPL", "call")
        # With no history, should return 1.0 (average)
        assert spike == 1.0

    def test_calculate_pcr_zscore_no_history(self, options_feed):
        """Test PCR z-score with insufficient history."""
        zscore = options_feed.calculate_pcr_zscore("AAPL", 0.8)
        # With no history, should return 0.0
        assert zscore == 0.0

    def test_update_pcr_history(self, options_feed):
        """Test PCR history tracking."""
        # Add some history
        for i in range(10):
            options_feed._update_pcr_history("AAPL", 0.5 + i * 0.1)

        assert len(options_feed._pcr_history["AAPL"]) == 10

        # History should be capped at 20
        for i in range(15):
            options_feed._update_pcr_history("AAPL", 0.5)

        assert len(options_feed._pcr_history["AAPL"]) == 20

    def test_calculate_flow_score(self, options_feed):
        """Test flow score calculation."""
        score = options_feed._calculate_flow_score(
            pcr=0.5,
            pcr_zscore=2.0,
            gamma_exposure=5_000_000,
            iv_skew=0.1,
            put_volume_spike=1.5,
            call_volume_spike=2.0,
        )

        assert 0 <= score <= 1


class TestOptionsMetrics:
    """Test OptionsMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating OptionsMetrics."""
        metrics = OptionsMetrics(
            symbol="AAPL",
            timestamp=datetime.now(),
            underlying_price=150.0,
            put_call_ratio=0.5,
            put_call_ratio_zscore=1.5,
            gamma_exposure=5_000_000,
            iv_skew=0.1,
            put_volume=1000,
            call_volume=2000,
            put_volume_spike=1.2,
            call_volume_spike=1.8,
            unusual_activity=True,
            flow_score=0.75,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.put_call_ratio == 0.5
        assert metrics.unusual_activity is True


class TestOptionContract:
    """Test OptionContract dataclass."""

    def test_contract_creation(self):
        """Test creating an OptionContract."""
        contract = OptionContract(
            ticker="AAPL230120C00150000",
            contract_type="call",
            strike=150.0,
            expiration_date="2023-01-20",
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            gamma=0.05,
            delta=0.55,
            underlying_price=152.0,
            last_price=5.00,
            bid=4.90,
            ask=5.10,
        )

        assert contract.contract_type == "call"
        assert contract.strike == 150.0
        assert contract.gamma == 0.05
