"""
Options Flow Trading Strategy.

Generates buy/sell signals based on:
- Put/Call Ratio (PCR) and z-score
- Gamma Exposure (GEX)
- IV Skew
- Volume spikes

Per spec: PCR + Gamma + IV Skew are visible in public data and
precede institutional hedging activity.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from swing.config import get_settings
from swing.data.options_feed import OptionsMetrics

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trading direction."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class Signal:
    """Trading signal with confidence and components."""

    symbol: str
    direction: SignalDirection
    confidence: float  # 0-1, minimum 0.60 to trade
    expiry_days: int
    entry_level: str  # "market" or specific price
    timestamp: datetime

    # Component breakdown
    components: Dict[str, float] = field(default_factory=dict)

    # Raw metrics for reference
    metrics: Optional[OptionsMetrics] = None

    def is_actionable(self, min_confidence: Optional[float] = None) -> bool:
        """Check if signal meets minimum confidence threshold."""
        settings = get_settings()
        threshold = min_confidence or settings.min_confidence
        return self.direction != SignalDirection.NONE and self.confidence >= threshold


class OptionsFlowStrategy:
    """
    Options flow-based swing trading strategy.

    Signal Rules (from spec):

    BULLISH (Go Long):
    - put_call_ratio < 0.5 (lots of calls relative to puts)
    - put_volume_spike < 1.0 (puts not spiking)
    - call_volume_spike > 1.5 (calls are active)
    - gamma_exposure > 1,000,000 (gamma positive/bullish)

    BEARISH (Go Short):
    - put_call_ratio > 1.5 (lots of puts relative to calls)
    - put_volume_spike > 2.0 (puts spiking hard)
    - gamma_exposure < -1,000,000 (gamma negative/bearish)
    - iv_skew > 0.20 (puts more expensive than calls)
    """

    # Thresholds (from spec)
    BULLISH_PCR_THRESHOLD = 0.5
    BEARISH_PCR_THRESHOLD = 1.5
    CALL_SPIKE_THRESHOLD = 1.5
    PUT_SPIKE_THRESHOLD = 2.0
    GAMMA_BULLISH_THRESHOLD = 1_000_000
    GAMMA_BEARISH_THRESHOLD = -1_000_000
    IV_SKEW_BEARISH_THRESHOLD = 0.20

    # Signal weights (from spec)
    WEIGHT_PCR = 0.40
    WEIGHT_GAMMA = 0.35
    WEIGHT_VOLUME = 0.15
    WEIGHT_IV = 0.10

    def __init__(self, expiry_days: Optional[int] = None):
        settings = get_settings()
        self.expiry_days = expiry_days or settings.signal_expiry_days

    def _normalize_pcr_signal(self, pcr: float, pcr_zscore: float, direction: SignalDirection) -> float:
        """
        Normalize PCR to 0-1 signal strength.

        For bullish: lower PCR = stronger signal
        For bearish: higher PCR = stronger signal
        """
        if direction == SignalDirection.LONG:
            # PCR < 0.5 is bullish, lower is better
            if pcr >= self.BULLISH_PCR_THRESHOLD:
                return 0.0
            # Scale: 0.0 -> 1.0, 0.5 -> 0.0
            raw_score = 1.0 - (pcr / self.BULLISH_PCR_THRESHOLD)
            # Boost with z-score (negative z-score = bullish)
            zscore_boost = max(0, -pcr_zscore / 3.0)
            return min(raw_score + zscore_boost * 0.2, 1.0)

        elif direction == SignalDirection.SHORT:
            # PCR > 1.5 is bearish, higher is better
            if pcr <= self.BEARISH_PCR_THRESHOLD:
                return 0.0
            # Scale: 1.5 -> 0.0, 3.0 -> 1.0
            raw_score = min((pcr - self.BEARISH_PCR_THRESHOLD) / 1.5, 1.0)
            # Boost with z-score (positive z-score = bearish)
            zscore_boost = max(0, pcr_zscore / 3.0)
            return min(raw_score + zscore_boost * 0.2, 1.0)

        return 0.0

    def _normalize_gamma_signal(self, gamma: float, direction: SignalDirection) -> float:
        """
        Normalize gamma exposure to 0-1 signal strength.

        For bullish: positive gamma = market makers will buy dips
        For bearish: negative gamma = market makers will sell rallies
        """
        if direction == SignalDirection.LONG:
            if gamma <= self.GAMMA_BULLISH_THRESHOLD:
                return 0.0
            # Scale: 1M -> 0.0, 10M -> 1.0
            return min((gamma - self.GAMMA_BULLISH_THRESHOLD) / 9_000_000, 1.0)

        elif direction == SignalDirection.SHORT:
            if gamma >= self.GAMMA_BEARISH_THRESHOLD:
                return 0.0
            # Scale: -1M -> 0.0, -10M -> 1.0
            return min(abs(gamma - self.GAMMA_BEARISH_THRESHOLD) / 9_000_000, 1.0)

        return 0.0

    def _normalize_volume_signal(
        self,
        put_spike: float,
        call_spike: float,
        direction: SignalDirection,
    ) -> float:
        """
        Normalize volume spike to 0-1 signal strength.

        For bullish: high call activity, low put activity
        For bearish: high put activity
        """
        if direction == SignalDirection.LONG:
            if call_spike < self.CALL_SPIKE_THRESHOLD or put_spike >= 1.0:
                return 0.0
            # Call spike strength
            call_score = min((call_spike - self.CALL_SPIKE_THRESHOLD) / 1.5, 1.0)
            # Put suppression bonus (lower is better)
            put_bonus = max(0, (1.0 - put_spike) / 1.0)
            return min(call_score + put_bonus * 0.3, 1.0)

        elif direction == SignalDirection.SHORT:
            if put_spike < self.PUT_SPIKE_THRESHOLD:
                return 0.0
            # Put spike strength
            return min((put_spike - self.PUT_SPIKE_THRESHOLD) / 2.0, 1.0)

        return 0.0

    def _normalize_iv_signal(self, iv_skew: float, direction: SignalDirection) -> float:
        """
        Normalize IV skew to 0-1 signal strength.

        Positive skew = puts more expensive = bearish sentiment
        Negative skew = calls more expensive = bullish sentiment
        """
        if direction == SignalDirection.LONG:
            # Negative skew is bullish
            if iv_skew >= 0:
                return 0.0
            return min(abs(iv_skew) / 0.3, 1.0)

        elif direction == SignalDirection.SHORT:
            # Positive skew > 0.20 is bearish
            if iv_skew < self.IV_SKEW_BEARISH_THRESHOLD:
                return 0.0
            return min((iv_skew - self.IV_SKEW_BEARISH_THRESHOLD) / 0.3, 1.0)

        return 0.0

    def _check_bullish_conditions(self, metrics: OptionsMetrics) -> bool:
        """Check if bullish conditions are met per spec."""
        return (
            metrics.put_call_ratio < self.BULLISH_PCR_THRESHOLD
            and metrics.put_volume_spike < 1.0
            and metrics.call_volume_spike > self.CALL_SPIKE_THRESHOLD
            and metrics.gamma_exposure > self.GAMMA_BULLISH_THRESHOLD
        )

    def _check_bearish_conditions(self, metrics: OptionsMetrics) -> bool:
        """Check if bearish conditions are met per spec."""
        return (
            metrics.put_call_ratio > self.BEARISH_PCR_THRESHOLD
            and metrics.put_volume_spike > self.PUT_SPIKE_THRESHOLD
            and metrics.gamma_exposure < self.GAMMA_BEARISH_THRESHOLD
            and metrics.iv_skew > self.IV_SKEW_BEARISH_THRESHOLD
        )

    def _calculate_signal_strength(
        self,
        metrics: OptionsMetrics,
        direction: SignalDirection,
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate weighted signal strength.

        Returns:
            Tuple of (confidence, component_scores)
        """
        pcr_signal = self._normalize_pcr_signal(
            metrics.put_call_ratio,
            metrics.put_call_ratio_zscore,
            direction,
        )

        gamma_signal = self._normalize_gamma_signal(
            metrics.gamma_exposure,
            direction,
        )

        volume_signal = self._normalize_volume_signal(
            metrics.put_volume_spike,
            metrics.call_volume_spike,
            direction,
        )

        iv_signal = self._normalize_iv_signal(
            metrics.iv_skew,
            direction,
        )

        # Weighted combination per spec
        confidence = (
            self.WEIGHT_PCR * pcr_signal
            + self.WEIGHT_GAMMA * gamma_signal
            + self.WEIGHT_VOLUME * volume_signal
            + self.WEIGHT_IV * iv_signal
        )

        components = {
            "pcr_signal": round(pcr_signal, 4),
            "gamma_signal": round(gamma_signal, 4),
            "volume_signal": round(volume_signal, 4),
            "iv_signal": round(iv_signal, 4),
        }

        return round(confidence, 4), components

    def generate_signal(self, metrics: OptionsMetrics) -> Signal:
        """
        Generate trading signal from options metrics.

        This is the main entry point for signal generation.
        """
        logger.info(f"Generating signal for {metrics.symbol}")

        # Determine direction
        if self._check_bullish_conditions(metrics):
            direction = SignalDirection.LONG
            logger.info(f"{metrics.symbol}: Bullish conditions met")

        elif self._check_bearish_conditions(metrics):
            direction = SignalDirection.SHORT
            logger.info(f"{metrics.symbol}: Bearish conditions met")

        else:
            # No clear signal - return neutral
            logger.debug(f"{metrics.symbol}: No clear signal")
            return Signal(
                symbol=metrics.symbol,
                direction=SignalDirection.NONE,
                confidence=0.0,
                expiry_days=self.expiry_days,
                entry_level="market",
                timestamp=datetime.now(),
                components={},
                metrics=metrics,
            )

        # Calculate signal strength
        confidence, components = self._calculate_signal_strength(metrics, direction)

        signal = Signal(
            symbol=metrics.symbol,
            direction=direction,
            confidence=confidence,
            expiry_days=self.expiry_days,
            entry_level="market",
            timestamp=datetime.now(),
            components=components,
            metrics=metrics,
        )

        logger.info(
            f"{metrics.symbol}: {direction.value} signal with confidence {confidence:.2%}"
        )

        return signal

    def generate_signals(self, metrics_list: List[OptionsMetrics]) -> List[Signal]:
        """Generate signals for multiple symbols."""
        signals = []
        for metrics in metrics_list:
            try:
                signal = self.generate_signal(metrics)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Failed to generate signal for {metrics.symbol}: {e}")
        return signals

    def get_actionable_signals(
        self,
        metrics_list: List[OptionsMetrics],
        min_confidence: Optional[float] = None,
    ) -> List[Signal]:
        """
        Generate signals and filter to only actionable ones.

        Returns signals with direction != NONE and confidence >= threshold.
        """
        settings = get_settings()
        threshold = min_confidence or settings.min_confidence

        signals = self.generate_signals(metrics_list)
        actionable = [s for s in signals if s.is_actionable(threshold)]

        # Sort by confidence descending
        actionable.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(
            f"Generated {len(signals)} signals, {len(actionable)} actionable "
            f"(confidence >= {threshold:.0%})"
        )

        return actionable

    def rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank signals by confidence, highest first."""
        return sorted(signals, key=lambda s: s.confidence, reverse=True)
