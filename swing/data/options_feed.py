"""
Options data feed from Polygon.io.

Fetches options chains and calculates derived metrics:
- Put/Call Ratio (PCR) with z-score
- Gamma Exposure (GEX)
- IV Skew
- Volume spikes
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from swing.config import get_settings
from swing.data.cache import DataCache

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Single option contract data."""

    ticker: str
    contract_type: str  # 'call' or 'put'
    strike: float
    expiration_date: str
    volume: int
    open_interest: int
    implied_volatility: float
    gamma: float
    delta: float
    underlying_price: float
    last_price: float
    bid: float
    ask: float


@dataclass
class OptionsMetrics:
    """Calculated options flow metrics for a symbol."""

    symbol: str
    timestamp: datetime
    underlying_price: float

    # Core metrics
    put_call_ratio: float
    put_call_ratio_zscore: float  # How unusual is this?
    gamma_exposure: float  # Net notional gamma
    iv_skew: float  # (IV_put - IV_call) / avg_IV

    # Volume analysis
    put_volume: int
    call_volume: int
    put_volume_spike: float  # vs 20-day avg
    call_volume_spike: float  # vs 20-day avg

    # Derived
    unusual_activity: bool
    flow_score: float  # 0-1 signal strength

    # Component breakdown
    components: Dict[str, float] = field(default_factory=dict)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0.0

    def wait(self) -> None:
        """Block until rate limit allows next call."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


class OptionsFeed:
    """Polygon.io options data pipeline."""

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[DataCache] = None,
        rate_limit: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.polygon_api_key
        self.cache = cache or DataCache()
        self.rate_limiter = RateLimiter(rate_limit or settings.polygon_rate_limit)

        # Historical PCR data for z-score calculation
        self._pcr_history: Dict[str, List[float]] = {}

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated API request with rate limiting."""
        self.rate_limiter.wait()

        params = params or {}
        params["apiKey"] = self.api_key

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_underlying_price(self, symbol: str) -> float:
        """Get current price for underlying stock."""
        cache_key = f"price_{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        data = self._make_request(endpoint)

        if data.get("results"):
            price = data["results"][0]["c"]  # Close price
            self.cache.set(cache_key, price, ttl_seconds=300)  # Cache 5 min
            return price

        raise ValueError(f"Could not get price for {symbol}")

    def get_options_chain(
        self,
        symbol: str,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
    ) -> List[OptionContract]:
        """
        Fetch options chain for a symbol.

        Args:
            symbol: Underlying ticker symbol
            expiration_date_gte: Filter expirations >= this date (YYYY-MM-DD)
            expiration_date_lte: Filter expirations <= this date (YYYY-MM-DD)

        Returns:
            List of OptionContract objects
        """
        cache_key = f"chain_{symbol}_{expiration_date_gte}_{expiration_date_lte}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return [OptionContract(**c) for c in cached]

        # Default: options expiring in next 30 days
        if expiration_date_gte is None:
            expiration_date_gte = datetime.now().strftime("%Y-%m-%d")
        if expiration_date_lte is None:
            expiration_date_lte = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        underlying_price = self.get_underlying_price(symbol)

        endpoint = f"/v3/reference/options/contracts"
        params = {
            "underlying_ticker": symbol,
            "expiration_date.gte": expiration_date_gte,
            "expiration_date.lte": expiration_date_lte,
            "limit": 250,
        }

        contracts = []
        data = self._make_request(endpoint, params)

        for result in data.get("results", []):
            # Get snapshot for each contract to get greeks and volume
            contract_ticker = result["ticker"]
            snapshot = self._get_contract_snapshot(contract_ticker)

            if snapshot:
                contracts.append(
                    OptionContract(
                        ticker=contract_ticker,
                        contract_type=result["contract_type"],
                        strike=result["strike_price"],
                        expiration_date=result["expiration_date"],
                        volume=snapshot.get("day", {}).get("volume", 0),
                        open_interest=snapshot.get("open_interest", 0),
                        implied_volatility=snapshot.get("implied_volatility", 0),
                        gamma=snapshot.get("greeks", {}).get("gamma", 0),
                        delta=snapshot.get("greeks", {}).get("delta", 0),
                        underlying_price=underlying_price,
                        last_price=snapshot.get("day", {}).get("close", 0),
                        bid=snapshot.get("last_quote", {}).get("bid", 0),
                        ask=snapshot.get("last_quote", {}).get("ask", 0),
                    )
                )

        # Cache the results
        self.cache.set(cache_key, [c.__dict__ for c in contracts])

        return contracts

    def _get_contract_snapshot(self, contract_ticker: str) -> Optional[Dict]:
        """Get snapshot data for a single options contract."""
        cache_key = f"snapshot_{contract_ticker}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        endpoint = f"/v3/snapshot/options/{contract_ticker}"

        try:
            data = self._make_request(endpoint)
            result = data.get("results")
            if result:
                self.cache.set(cache_key, result, ttl_seconds=900)  # 15 min cache
            return result
        except Exception as e:
            logger.warning(f"Failed to get snapshot for {contract_ticker}: {e}")
            return None

    def calculate_put_call_ratio(self, options_chain: List[OptionContract]) -> tuple[float, int, int]:
        """
        Calculate put/call ratio from options chain.

        Returns:
            Tuple of (ratio, put_volume, call_volume)
        """
        put_volume = sum(c.volume for c in options_chain if c.contract_type == "put")
        call_volume = sum(c.volume for c in options_chain if c.contract_type == "call")

        ratio = put_volume / call_volume if call_volume > 0 else 0.0
        return ratio, put_volume, call_volume

    def calculate_gamma_exposure(self, options_chain: List[OptionContract]) -> float:
        """
        Calculate net gamma exposure (GEX).

        GEX = sum(gamma * open_interest * underlying_price^2 * 100)
        Positive = calls dominating (market makers short gamma, will buy dips)
        Negative = puts dominating (market makers long gamma, will sell rallies)
        """
        gex = 0.0
        for contract in options_chain:
            # Calls contribute positive gamma, puts contribute negative
            sign = 1 if contract.contract_type == "call" else -1
            gex += (
                sign
                * contract.gamma
                * contract.open_interest
                * (contract.underlying_price ** 2)
                * 100
            )
        return gex

    def calculate_iv_skew(self, options_chain: List[OptionContract]) -> float:
        """
        Calculate IV skew between puts and calls.

        IV Skew = (IV_put - IV_call) / avg_IV
        Positive skew = puts more expensive (bearish sentiment)
        Negative skew = calls more expensive (bullish sentiment)
        """
        if not options_chain:
            return 0.0

        underlying_price = options_chain[0].underlying_price

        # Find ATM options (closest to underlying price)
        atm_calls = [
            c for c in options_chain
            if c.contract_type == "call"
            and abs(c.strike - underlying_price) / underlying_price < 0.05
        ]
        atm_puts = [
            c for c in options_chain
            if c.contract_type == "put"
            and abs(c.strike - underlying_price) / underlying_price < 0.05
        ]

        if not atm_calls or not atm_puts:
            return 0.0

        # Average IV for ATM options
        avg_call_iv = sum(c.implied_volatility for c in atm_calls) / len(atm_calls)
        avg_put_iv = sum(c.implied_volatility for c in atm_puts) / len(atm_puts)
        avg_iv = (avg_call_iv + avg_put_iv) / 2

        if avg_iv <= 0:
            return 0.0

        return (avg_put_iv - avg_call_iv) / avg_iv

    def calculate_volume_spike(
        self,
        current_volume: int,
        symbol: str,
        option_type: str,
    ) -> float:
        """
        Calculate volume spike vs historical average.

        Returns ratio of current volume to 20-day average.
        > 1.0 means above average, < 1.0 means below average.
        """
        # For now, use a simple heuristic based on typical volumes
        # In production, you'd store and query historical data
        history_key = f"{symbol}_{option_type}_volume"
        history = self._pcr_history.get(history_key, [])

        if len(history) < 5:
            # Not enough history, assume current is average
            return 1.0

        avg_volume = sum(history) / len(history)
        if avg_volume <= 0:
            return 1.0

        return current_volume / avg_volume

    def _update_pcr_history(self, symbol: str, pcr: float) -> None:
        """Update PCR history for z-score calculation."""
        if symbol not in self._pcr_history:
            self._pcr_history[symbol] = []

        self._pcr_history[symbol].append(pcr)

        # Keep last 20 data points
        if len(self._pcr_history[symbol]) > 20:
            self._pcr_history[symbol] = self._pcr_history[symbol][-20:]

    def calculate_pcr_zscore(self, symbol: str, current_pcr: float) -> float:
        """
        Calculate z-score of current PCR vs historical.

        Z-score shows how many standard deviations from mean.
        High positive z-score = unusually high put activity (bearish)
        High negative z-score = unusually low put activity (bullish)
        """
        history = self._pcr_history.get(symbol, [])

        if len(history) < 5:
            return 0.0

        import statistics

        mean = statistics.mean(history)
        stdev = statistics.stdev(history) if len(history) > 1 else 1.0

        if stdev == 0:
            return 0.0

        return (current_pcr - mean) / stdev

    def get_options_metrics(self, symbol: str) -> OptionsMetrics:
        """
        Fetch options data and calculate all metrics for a symbol.

        This is the main entry point for the data pipeline.
        """
        logger.info(f"Fetching options metrics for {symbol}")

        # Get options chain
        options_chain = self.get_options_chain(symbol)

        if not options_chain:
            raise ValueError(f"No options data available for {symbol}")

        underlying_price = options_chain[0].underlying_price

        # Calculate core metrics
        pcr, put_volume, call_volume = self.calculate_put_call_ratio(options_chain)
        self._update_pcr_history(symbol, pcr)
        pcr_zscore = self.calculate_pcr_zscore(symbol, pcr)

        gamma_exposure = self.calculate_gamma_exposure(options_chain)
        iv_skew = self.calculate_iv_skew(options_chain)

        put_volume_spike = self.calculate_volume_spike(put_volume, symbol, "put")
        call_volume_spike = self.calculate_volume_spike(call_volume, symbol, "call")

        # Determine if unusual activity
        unusual_activity = (
            abs(pcr_zscore) > 1.5
            or put_volume_spike > 2.0
            or call_volume_spike > 2.0
        )

        # Calculate flow score (0-1 signal strength)
        flow_score = self._calculate_flow_score(
            pcr=pcr,
            pcr_zscore=pcr_zscore,
            gamma_exposure=gamma_exposure,
            iv_skew=iv_skew,
            put_volume_spike=put_volume_spike,
            call_volume_spike=call_volume_spike,
        )

        return OptionsMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            underlying_price=underlying_price,
            put_call_ratio=pcr,
            put_call_ratio_zscore=pcr_zscore,
            gamma_exposure=gamma_exposure,
            iv_skew=iv_skew,
            put_volume=put_volume,
            call_volume=call_volume,
            put_volume_spike=put_volume_spike,
            call_volume_spike=call_volume_spike,
            unusual_activity=unusual_activity,
            flow_score=flow_score,
            components={
                "pcr_raw": pcr,
                "pcr_zscore": pcr_zscore,
                "gamma": gamma_exposure,
                "iv_skew": iv_skew,
                "put_spike": put_volume_spike,
                "call_spike": call_volume_spike,
            },
        )

    def _calculate_flow_score(
        self,
        pcr: float,
        pcr_zscore: float,
        gamma_exposure: float,
        iv_skew: float,
        put_volume_spike: float,
        call_volume_spike: float,
    ) -> float:
        """
        Calculate combined flow score (0-1).

        Higher score = stronger signal (direction determined separately).
        """
        # Normalize each component to 0-1 range
        pcr_score = min(abs(pcr_zscore) / 3.0, 1.0)  # 3 std dev = max

        # Gamma: normalize to millions
        gamma_score = min(abs(gamma_exposure) / 10_000_000, 1.0)

        # IV skew: typically -0.5 to 0.5
        iv_score = min(abs(iv_skew) / 0.5, 1.0)

        # Volume spikes
        volume_score = min(max(put_volume_spike, call_volume_spike) / 3.0, 1.0)

        # Weighted combination (per spec)
        flow_score = (
            0.40 * pcr_score
            + 0.35 * gamma_score
            + 0.15 * volume_score
            + 0.10 * iv_score
        )

        return round(flow_score, 4)

    def get_metrics_for_symbols(self, symbols: List[str]) -> Dict[str, OptionsMetrics]:
        """Fetch metrics for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_options_metrics(symbol)
            except Exception as e:
                logger.error(f"Failed to get metrics for {symbol}: {e}")
        return results


# CLI entry point for testing
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch options metrics")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Symbol to fetch")
    parser.add_argument("--days", type=int, default=20, help="Lookback days")
    args = parser.parse_args()

    feed = OptionsFeed()
    metrics = feed.get_options_metrics(args.symbol)

    print(json.dumps({
        "symbol": metrics.symbol,
        "timestamp": metrics.timestamp.isoformat(),
        "underlying_price": metrics.underlying_price,
        "put_call_ratio": metrics.put_call_ratio,
        "put_call_ratio_zscore": metrics.put_call_ratio_zscore,
        "gamma_exposure": metrics.gamma_exposure,
        "iv_skew": metrics.iv_skew,
        "put_volume": metrics.put_volume,
        "call_volume": metrics.call_volume,
        "unusual_activity": metrics.unusual_activity,
        "flow_score": metrics.flow_score,
    }, indent=2))
