"""
Alpaca Broker API integration.

Handles order placement, fill tracking, and account management.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from swing.config import get_settings

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status tracking."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order tracking."""

    order_id: str
    symbol: str
    quantity: int
    direction: str  # 'BUY' or 'SELL'
    order_type: str  # 'market' or 'limit'
    limit_price: Optional[float]
    submitted_at: datetime
    status: OrderStatus = OrderStatus.PENDING

    # Fill info (populated after fill)
    fill_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    actual_slippage: float = 0.0

    # Metadata
    time_in_force: str = "gtc"  # good til canceled


class AlpacaExecutor:
    """
    Alpaca broker execution.

    Uses limit orders by default to control slippage.
    Per spec: 0.5% worse than market to guarantee fill.
    """

    # Default slippage buffer for limit orders
    LIMIT_BUFFER_PCT = 0.005  # 0.5%

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True,
    ):
        settings = get_settings()

        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key
        self.base_url = base_url or settings.alpaca_base_url
        self.paper = paper

        self._client = None
        self.orders: Dict[str, Order] = {}

    def _get_client(self):
        """Lazy init of Alpaca client."""
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient

                self._client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=self.paper,
                )
            except ImportError:
                raise ImportError(
                    "alpaca-py not installed. Run: pip install alpaca-py"
                )
        return self._client

    def get_account(self) -> Dict:
        """Get account information."""
        client = self._get_client()
        account = client.get_account()

        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    def get_last_price(self, symbol: str) -> float:
        """Get last trade price for symbol."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest

            data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trades = data_client.get_stock_latest_trade(request)

            return float(trades[symbol].price)

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            raise

    def place_order(
        self,
        symbol: str,
        quantity: int,
        direction: str,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "gtc",
    ) -> Order:
        """
        Place an order.

        Per spec: Use limit orders 0.5% worse than market to control slippage.
        """
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = self._get_client()

        # Determine side
        side = OrderSide.BUY if direction.upper() == "BUY" else OrderSide.SELL

        # Determine time in force
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.GTC)

        # Calculate limit price if not provided
        if order_type == "limit" and limit_price is None:
            current_price = self.get_last_price(symbol)
            if direction.upper() == "BUY":
                limit_price = current_price * (1 + self.LIMIT_BUFFER_PCT)
            else:
                limit_price = current_price * (1 - self.LIMIT_BUFFER_PCT)
            limit_price = round(limit_price, 2)

        # Create order request
        if order_type == "limit":
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=tif,
                limit_price=limit_price,
            )
        else:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=tif,
            )

        # Submit order
        try:
            alpaca_order = client.submit_order(order_request)

            order = Order(
                order_id=str(alpaca_order.id),
                symbol=symbol,
                quantity=quantity,
                direction=direction.upper(),
                order_type=order_type,
                limit_price=limit_price,
                submitted_at=datetime.now(),
                time_in_force=time_in_force,
            )

            self.orders[order.order_id] = order

            logger.info(
                f"Placed {direction} {order_type} order: {quantity} {symbol} "
                f"@ ${limit_price or 'market'}"
            )

            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order."""
        client = self._get_client()

        try:
            alpaca_order = client.get_order_by_id(order_id)

            status_map = {
                "new": OrderStatus.PENDING,
                "accepted": OrderStatus.PENDING,
                "pending_new": OrderStatus.PENDING,
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "canceled": OrderStatus.CANCELED,
                "expired": OrderStatus.CANCELED,
                "rejected": OrderStatus.REJECTED,
            }

            return status_map.get(alpaca_order.status.value, OrderStatus.PENDING)

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return OrderStatus.PENDING

    def get_fill_price(self, order_id: str) -> Optional[float]:
        """Get fill price for a filled order."""
        client = self._get_client()

        try:
            alpaca_order = client.get_order_by_id(order_id)

            if alpaca_order.filled_avg_price:
                return float(alpaca_order.filled_avg_price)
            return None

        except Exception as e:
            logger.error(f"Failed to get fill price: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        client = self._get_client()

        try:
            client.cancel_order_by_id(order_id)

            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELED

            logger.info(f"Canceled order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def check_fills(self) -> List[Order]:
        """
        Poll for fills on pending orders.

        Returns list of orders that were filled.
        """
        filled_orders = []

        for order_id, order in list(self.orders.items()):
            if order.status != OrderStatus.PENDING:
                continue

            status = self.get_order_status(order_id)

            if status == OrderStatus.FILLED:
                fill_price = self.get_fill_price(order_id)

                order.status = OrderStatus.FILLED
                order.fill_price = fill_price
                order.filled_at = datetime.now()

                if order.limit_price and fill_price:
                    order.actual_slippage = abs(fill_price - order.limit_price)

                filled_orders.append(order)

                logger.info(
                    f"Order filled: {order.quantity} {order.symbol} @ ${fill_price:.2f}"
                )

            elif status in (OrderStatus.CANCELED, OrderStatus.REJECTED):
                order.status = status

            # Auto-cancel orders older than 1 hour
            elif (datetime.now() - order.submitted_at).seconds > 3600:
                self.cancel_order(order_id)

        return filled_orders

    def get_positions(self) -> Dict[str, Dict]:
        """Get all open positions."""
        client = self._get_client()

        try:
            positions = client.get_all_positions()

            return {
                pos.symbol: {
                    "qty": int(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "side": pos.side.value,
                }
                for pos in positions
            }

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position in a symbol."""
        client = self._get_client()

        try:
            client.close_position(symbol)
            logger.info(f"Closed position in {symbol}")

            # Return a synthetic order for tracking
            return Order(
                order_id=f"close_{symbol}_{int(time.time())}",
                symbol=symbol,
                quantity=0,  # Will be determined by position size
                direction="SELL",
                order_type="market",
                limit_price=None,
                submitted_at=datetime.now(),
                status=OrderStatus.FILLED,
            )

        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return None

    def close_all_positions(self) -> bool:
        """Close all open positions."""
        client = self._get_client()

        try:
            client.close_all_positions(cancel_orders=True)
            logger.info("Closed all positions")
            return True

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False
