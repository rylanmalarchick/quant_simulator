"""
Stock price feed using Yahoo Finance as fallback.

Provides OHLCV data for backtesting and current prices.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

from swing.data.cache import DataCache

logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """Single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockFeed:
    """Yahoo Finance stock data feed."""

    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache or DataCache()

    def get_current_price(self, symbol: str) -> float:
        """Get current/latest price for a symbol."""
        cache_key = f"stock_price_{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")

        if data.empty:
            raise ValueError(f"No price data for {symbol}")

        price = float(data["Close"].iloc[-1])
        self.cache.set(cache_key, price, ttl_seconds=300)  # 5 min cache

        return price

    def get_historical(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "1mo",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Ticker symbol
            start_date: Start date (optional, uses period if not set)
            end_date: End date (optional, defaults to today)
            period: Period string like "1mo", "3mo", "1y" (used if start_date not set)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = f"hist_{symbol}_{start_date}_{end_date}_{period}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            df = pd.DataFrame(cached)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
                df = df.set_index('Date')
                # Normalize index to date only (remove time component)
                df.index = df.index.normalize()
            return df

        ticker = yf.Ticker(symbol)

        if start_date:
            # Convert datetime to string format for yfinance
            start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if end_date and isinstance(end_date, datetime) else None
            data = ticker.history(start=start_str, end=end_str)
        else:
            data = ticker.history(period=period)

        if data.empty:
            raise ValueError(f"No historical data for {symbol}")

        # Get the relevant columns and normalize index to naive datetime
        result = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        # Remove timezone info and normalize to date only
        if result.index.tz is not None:
            result.index = result.index.tz_convert(None)
        result.index = result.index.normalize()
        
        # Cache as dict for JSON serialization (convert index to string)
        cache_data = result.reset_index()
        cache_data.columns = ['Date'] + list(cache_data.columns[1:])
        cache_data['Date'] = cache_data['Date'].astype(str)
        self.cache.set(
            cache_key,
            cache_data.to_dict("records"),
            ttl_seconds=3600,  # 1 hour cache
        )

        return result

    def get_ohlcv_bars(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[OHLCV]:
        """
        Get OHLCV bars as list of dataclass objects.

        Args:
            symbol: Ticker symbol
            days: Number of days of history

        Returns:
            List of OHLCV bars
        """
        start_date = datetime.now() - timedelta(days=days)
        df = self.get_historical(symbol, start_date=start_date)

        bars = []
        for idx, row in df.iterrows():
            bars.append(
                OHLCV(
                    timestamp=idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
            )

        return bars

    def calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """
        Calculate annualized volatility.

        Args:
            symbol: Ticker symbol
            window: Rolling window in days

        Returns:
            Annualized volatility as decimal (0.25 = 25%)
        """
        df = self.get_historical(symbol, period="3mo")

        if len(df) < window:
            raise ValueError(f"Not enough data for {window}-day volatility")

        # Daily returns
        returns = df["Close"].pct_change().dropna()

        # Rolling standard deviation, annualized
        daily_vol = returns.tail(window).std()
        annual_vol = daily_vol * (252 ** 0.5)

        return float(annual_vol)

    def get_average_volume(self, symbol: str, days: int = 20) -> float:
        """Get average daily volume over period."""
        df = self.get_historical(symbol, period=f"{days}d")
        return float(df["Volume"].mean())

    def has_earnings_soon(self, symbol: str, days: int = 3) -> bool:
        """
        Check if stock has earnings announcement within N days.

        Returns True if earnings are coming up (should avoid trading).
        """
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is None or calendar.empty:
                return False

            # Check earnings date
            if "Earnings Date" in calendar.index:
                earnings_dates = calendar.loc["Earnings Date"]
                if not isinstance(earnings_dates, pd.Series):
                    earnings_dates = [earnings_dates]

                for date in earnings_dates:
                    if isinstance(date, str):
                        date = datetime.strptime(date, "%Y-%m-%d")
                    if hasattr(date, "to_pydatetime"):
                        date = date.to_pydatetime()

                    days_until = (date - datetime.now()).days
                    if 0 <= days_until <= days:
                        logger.info(f"{symbol} has earnings in {days_until} days")
                        return True

            return False

        except Exception as e:
            logger.warning(f"Could not check earnings for {symbol}: {e}")
            return False  # Assume no earnings if can't check
