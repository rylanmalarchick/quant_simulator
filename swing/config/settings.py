"""
Settings module for swing trading system.

Loads configuration from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Immutable configuration settings for the trading system."""

    # API Keys
    polygon_api_key: str
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str

    # Capital & Risk Parameters
    initial_capital: float
    risk_per_trade: float  # e.g., 0.02 = 2%
    max_positions: int
    max_daily_loss: float  # e.g., 0.05 = 5%
    max_position_pct: float  # e.g., 0.20 = 20% of capital

    # Trading Parameters
    default_stop_loss_pct: float  # e.g., 0.02 = 2%
    profit_target_pct: float  # e.g., 0.10 = 10%
    signal_expiry_days: int
    min_confidence: float  # Minimum signal confidence to trade

    # Symbols to Watch
    symbols: List[str]

    # Alerting
    alert_email: str
    alert_slack_webhook: str

    # Cache Settings
    cache_dir: Path
    cache_ttl_seconds: int

    # Rate Limiting (Polygon free tier: 5 calls/min)
    polygon_rate_limit: int  # calls per minute

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        load_dotenv()

        symbols_str = os.getenv("SYMBOLS_TO_WATCH", "AAPL,TSLA,NVDA,AMD,PLTR,GOOGL,MSFT")
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

        cache_dir = Path(os.getenv("CACHE_DIR", ".cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            # API Keys
            polygon_api_key=os.getenv("POLYGON_API_KEY", ""),
            alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            alpaca_base_url=os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
            # Capital & Risk
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "25000")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.02")),
            max_positions=int(os.getenv("MAX_POSITIONS", "5")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.05")),
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.20")),
            # Trading Parameters
            default_stop_loss_pct=float(os.getenv("DEFAULT_STOP_LOSS_PCT", "0.02")),
            profit_target_pct=float(os.getenv("PROFIT_TARGET_PCT", "0.10")),
            signal_expiry_days=int(os.getenv("SIGNAL_EXPIRY_DAYS", "5")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.60")),
            # Symbols
            symbols=symbols,
            # Alerting
            alert_email=os.getenv("ALERT_EMAIL", ""),
            alert_slack_webhook=os.getenv("ALERT_SLACK_WEBHOOK", ""),
            # Cache
            cache_dir=cache_dir,
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "7200")),  # 2 hours
            # Rate Limiting
            polygon_rate_limit=int(os.getenv("POLYGON_RATE_LIMIT", "5")),
        )

    def validate(self) -> List[str]:
        """Validate settings and return list of errors."""
        errors = []

        if not self.polygon_api_key:
            errors.append("POLYGON_API_KEY is required")
        if not self.alpaca_api_key:
            errors.append("ALPACA_API_KEY is required")
        if not self.alpaca_secret_key:
            errors.append("ALPACA_SECRET_KEY is required")

        if self.initial_capital <= 0:
            errors.append("INITIAL_CAPITAL must be positive")
        if not 0 < self.risk_per_trade <= 0.10:
            errors.append("RISK_PER_TRADE should be between 0 and 10%")
        if not 0 < self.max_daily_loss <= 0.20:
            errors.append("MAX_DAILY_LOSS should be between 0 and 20%")
        if self.max_positions < 1:
            errors.append("MAX_POSITIONS must be at least 1")
        if not self.symbols:
            errors.append("At least one symbol required in SYMBOLS_TO_WATCH")

        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
