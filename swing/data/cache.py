"""
Local caching module to reduce API calls.

Implements file-based caching with TTL for options and stock data.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from swing.config import get_settings


@dataclass
class CacheEntry:
    """Cached data with metadata."""

    data: Any
    timestamp: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > self.timestamp + self.ttl_seconds


class DataCache:
    """File-based cache for API responses."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_seconds: Optional[int] = None):
        settings = get_settings()
        self.cache_dir = cache_dir or settings.cache_dir
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                entry_data = json.load(f)

            entry = CacheEntry(
                data=entry_data["data"],
                timestamp=entry_data["timestamp"],
                ttl_seconds=entry_data["ttl_seconds"],
            )

            if entry.is_expired():
                cache_path.unlink(missing_ok=True)
                return None

            return entry.data

        except (json.JSONDecodeError, KeyError):
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Cache a value with TTL."""
        cache_path = self._get_cache_path(key)
        ttl = ttl_seconds or self.ttl_seconds

        entry = CacheEntry(data=value, timestamp=time.time(), ttl_seconds=ttl)

        with open(cache_path, "w") as f:
            json.dump(asdict(entry), f)

    def invalidate(self, key: str) -> None:
        """Remove a cached entry."""
        cache_path = self._get_cache_path(key)
        cache_path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink(missing_ok=True)

    def clear_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    entry_data = json.load(f)
                entry = CacheEntry(
                    data=entry_data["data"],
                    timestamp=entry_data["timestamp"],
                    ttl_seconds=entry_data["ttl_seconds"],
                )
                if entry.is_expired():
                    cache_file.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError):
                cache_file.unlink(missing_ok=True)
                removed += 1
        return removed
