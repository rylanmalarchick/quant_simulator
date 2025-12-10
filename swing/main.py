"""
Main trading loop for swing trading system.

Runs the full pipeline:
1. Fetch options data
2. Generate signals
3. Check for exits on open positions
4. Open new positions
5. Monitor and alert
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from swing.config import get_settings
from swing.data.options_feed import OptionsFeed
from swing.data.stock_feed import StockFeed
from swing.execution.alpaca_api import AlpacaExecutor
from swing.monitoring.alerts import AlertManager
from swing.monitoring.trade_monitor import TradeMonitor
from swing.risk.position_manager import PositionManager
from swing.signals.options_flow_strategy import OptionsFlowStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SwingTrader:
    """Main trading orchestrator."""

    def __init__(self, paper: bool = True):
        self.settings = get_settings()

        # Initialize components
        self.options_feed = OptionsFeed()
        self.stock_feed = StockFeed()
        self.strategy = OptionsFlowStrategy()
        self.position_manager = PositionManager()
        self.executor = AlpacaExecutor(paper=paper)
        self.monitor = TradeMonitor()
        self.alerts = AlertManager()

        self.paper = paper
        self.running = False

    def run_once(self) -> None:
        """Run a single iteration of the trading loop."""
        logger.info("Running trading iteration...")

        try:
            # 1. Fetch options data for all symbols
            symbols = self.settings.symbols
            metrics_dict = self.options_feed.get_metrics_for_symbols(symbols)

            if not metrics_dict:
                logger.warning("No options data available")
                return

            # 2. Generate signals
            metrics_list = list(metrics_dict.values())
            signals = self.strategy.get_actionable_signals(metrics_list)

            logger.info(f"Generated {len(signals)} actionable signals")

            for signal in signals:
                self.alerts.signal_alert(
                    signal.symbol, signal.direction.value, signal.confidence
                )

            # 3. Check for exits on open positions
            for symbol in list(self.position_manager.positions.keys()):
                try:
                    current_price = self.stock_feed.get_current_price(symbol)
                    exit_reason = self.position_manager.should_exit(symbol, current_price)

                    if exit_reason:
                        # Close the position
                        self.executor.close_position(symbol)
                        trade = self.position_manager.close_position(
                            symbol, current_price, exit_reason
                        )

                        if trade:
                            # Record in monitor
                            pos = self.position_manager.positions.get(symbol)
                            if pos:
                                self.monitor.record_trade(
                                    symbol=symbol,
                                    direction=trade["direction"],
                                    entry_price=trade["entry_price"],
                                    exit_price=trade["exit_price"],
                                    quantity=trade["shares"],
                                    entry_time=datetime.fromisoformat(trade["entry_time"]),
                                    exit_time=datetime.now(),
                                    exit_reason=exit_reason.value,
                                )

                            # Send alerts based on exit reason
                            if exit_reason.value == "STOP_HIT":
                                self.alerts.stop_loss_alert(
                                    symbol,
                                    trade["entry_price"],
                                    trade["exit_price"],
                                    trade["realized_pnl"],
                                )
                            elif exit_reason.value == "PROFIT_TARGET":
                                self.alerts.profit_target_alert(
                                    symbol,
                                    trade["entry_price"],
                                    trade["exit_price"],
                                    trade["realized_pnl"],
                                )

                except Exception as e:
                    logger.error(f"Error checking exit for {symbol}: {e}")

            # 4. Open new positions from signals
            for signal in signals:
                if signal.symbol in self.position_manager.positions:
                    continue

                can_trade, reason = self.position_manager.can_take_trade(signal.symbol)
                if not can_trade:
                    logger.info(f"Cannot trade {signal.symbol}: {reason}")
                    continue

                # Check for upcoming earnings
                if self.stock_feed.has_earnings_soon(signal.symbol, days=3):
                    logger.info(f"Skipping {signal.symbol}: earnings coming up")
                    continue

                try:
                    # Get current price
                    current_price = self.stock_feed.get_current_price(signal.symbol)

                    # Open position
                    position = self.position_manager.open_position(signal, current_price)

                    if position:
                        # Place order
                        order = self.executor.place_order(
                            symbol=signal.symbol,
                            quantity=position.shares,
                            direction="BUY" if signal.direction.value == "LONG" else "SELL",
                            order_type="limit",
                        )

                        self.alerts.position_opened_alert(
                            signal.symbol,
                            signal.direction.value,
                            position.shares,
                            current_price,
                        )

                except Exception as e:
                    logger.error(f"Error opening position for {signal.symbol}: {e}")

            # 5. Update monitoring
            account = self.executor.get_account()
            self.monitor.update_realtime(account["equity"])

            # Log summary
            metrics = self.monitor.get_metrics()
            logger.info(
                f"Portfolio: ${account['equity']:.2f} | "
                f"Positions: {len(self.position_manager.positions)} | "
                f"Daily P&L: ${metrics.get('daily_pnl', 0):.2f}"
            )

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}", exc_info=True)
            self.alerts.send_alert(
                self.alerts.AlertType.SYSTEM_ERROR,
                "Trading Error",
                str(e),
            )

    def run(self, interval_minutes: int = 120) -> None:
        """
        Run the trading loop continuously.

        Per spec: Update every 2 hours during market hours.
        """
        self.running = True
        logger.info(f"Starting trading loop (interval: {interval_minutes} min)")

        while self.running:
            # Check if market is open (simple check)
            now = datetime.now()
            hour = now.hour

            # Market hours: 9:30 AM - 4:00 PM ET (simplified)
            if 9 <= hour <= 16 and now.weekday() < 5:
                self.run_once()
            else:
                logger.info("Market closed, skipping iteration")

            # Wait for next iteration
            time.sleep(interval_minutes * 60)

    def stop(self) -> None:
        """Stop the trading loop."""
        self.running = False
        logger.info("Trading loop stopped")

    def status(self) -> dict:
        """Get current status."""
        return {
            "running": self.running,
            "paper": self.paper,
            "portfolio": self.position_manager.get_portfolio_summary(),
            "metrics": self.monitor.get_metrics(),
        }


def main():
    parser = argparse.ArgumentParser(description="Swing Trading System")
    parser.add_argument(
        "--mode",
        choices=["live", "paper"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=25000,
        help="Starting capital (default: 25000)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Update interval in minutes (default: 120)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )

    args = parser.parse_args()

    # Validate settings
    settings = get_settings()
    errors = settings.validate()
    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Create trader
    paper = args.mode == "paper"
    trader = SwingTrader(paper=paper)

    if paper:
        logger.info("Running in PAPER trading mode")
    else:
        logger.warning("Running in LIVE trading mode - real money!")

    try:
        if args.once:
            trader.run_once()
        else:
            trader.run(interval_minutes=args.interval)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        trader.stop()


if __name__ == "__main__":
    main()
