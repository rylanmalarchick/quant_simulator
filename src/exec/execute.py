"""
Execution module for quantSim.

Executes trading signals with proper position sizing based on:
- Account equity
- Risk per trade (configurable)
- Maximum position size limits
- Current portfolio exposure
"""

from src.signal.generate import generate_signals
from src.exec.alpaca_client import submit_order, get_alpaca_api
from src.exec.risk import check_risk_controls, calculate_position_size
from src.config import load_config
from src.logging import get_logger

logger = get_logger(__name__)


def get_account_equity() -> float:
    """Get current account equity from Alpaca."""
    try:
        api = get_alpaca_api()
        account = api.get_account()
        return float(account.equity)
    except Exception as e:
        logger.error(f"Failed to get account equity: {e}")
        return 0.0


def get_current_price(symbol: str) -> float:
    """Get current price for a symbol."""
    try:
        api = get_alpaca_api()
        # Get latest trade price
        trades = api.get_latest_trade(symbol)
        return float(trades.price)
    except Exception as e:
        logger.error(f"Failed to get price for {symbol}: {e}")
        return 0.0


def get_current_positions() -> dict:
    """Get current positions from Alpaca."""
    try:
        api = get_alpaca_api()
        positions = api.list_positions()
        return {p.symbol: float(p.qty) for p in positions}
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return {}


def calculate_daily_turnover(trades_today: list, account_equity: float) -> float:
    """Calculate daily turnover as percentage of account."""
    if account_equity <= 0:
        return 0.0
    total_traded = sum(abs(t.get('value', 0)) for t in trades_today)
    return total_traded / account_equity


def execute_signals():
    """
    Executes the trading signals with proper position sizing.
    
    Position sizing is based on:
    1. Account equity
    2. Risk per trade (from config, default 2%)
    3. Maximum position size (from config)
    4. Stop loss distance
    """
    logger.info("Executing signals...")
    config = load_config()
    signals = generate_signals()
    
    if not signals:
        logger.warning("No signals to execute.")
        return
    
    # Get account info
    account_equity = get_account_equity()
    if account_equity <= 0:
        logger.error("Cannot execute: account equity is zero or negative")
        return
    
    current_positions = get_current_positions()
    trades_executed = []
    
    # Get execution parameters from config
    risk_per_trade = config['execution'].get('risk_per_trade', 0.02)  # 2% default
    max_position_pct = config['execution'].get('max_position_size', 0.10)  # 10% max
    stop_loss_pct = config['execution'].get('stop_loss_pct', 0.05)  # 5% stop loss
    
    logger.info(f"Account equity: ${account_equity:,.2f}")
    logger.info(f"Risk per trade: {risk_per_trade:.1%}")
    logger.info(f"Max position size: {max_position_pct:.1%}")
    
    # Execute buy signals
    logger.info("Processing buy signals...")
    for _, row in signals['buy'].iterrows():
        symbol = row['symbol']
        
        # Skip if already in position
        if symbol in current_positions and current_positions[symbol] > 0:
            logger.info(f"Skipping {symbol}: already in position")
            continue
        
        # Get current price
        price = get_current_price(symbol)
        if price <= 0:
            logger.warning(f"Skipping {symbol}: could not get price")
            continue
        
        # Calculate position size
        shares = calculate_position_size(
            account_equity=account_equity,
            price=price,
            risk_per_trade=risk_per_trade,
            max_position_pct=max_position_pct,
            stop_loss_pct=stop_loss_pct
        )
        
        if shares <= 0:
            logger.warning(f"Skipping {symbol}: calculated position size is zero")
            continue
        
        # Calculate order value for risk check
        order_value = shares * price
        order_size_pct = order_value / account_equity
        daily_turnover = calculate_daily_turnover(trades_executed, account_equity)
        
        if check_risk_controls(order_size_pct, daily_turnover):
            order = submit_order(symbol, shares, 'buy')
            if order:
                trades_executed.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'shares': shares,
                    'value': order_value
                })
                logger.info(f"BUY {shares} {symbol} @ ~${price:.2f} (${order_value:,.2f})")
        else:
            logger.warning(f"Skipping {symbol}: failed risk controls")
    
    # Execute sell signals (close existing positions)
    logger.info("Processing sell signals...")
    for _, row in signals['sell'].iterrows():
        symbol = row['symbol']
        
        # Only sell if we have a position
        if symbol not in current_positions or current_positions[symbol] <= 0:
            logger.info(f"Skipping sell {symbol}: no position to close")
            continue
        
        shares = int(current_positions[symbol])
        price = get_current_price(symbol)
        order_value = shares * price if price > 0 else 0
        
        daily_turnover = calculate_daily_turnover(trades_executed, account_equity)
        order_size_pct = order_value / account_equity if account_equity > 0 else 0
        
        if check_risk_controls(order_size_pct, daily_turnover):
            order = submit_order(symbol, shares, 'sell')
            if order:
                trades_executed.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'shares': shares,
                    'value': order_value
                })
                logger.info(f"SELL {shares} {symbol} @ ~${price:.2f} (${order_value:,.2f})")
        else:
            logger.warning(f"Skipping sell {symbol}: failed risk controls")
    
    # Summary
    total_buys = len([t for t in trades_executed if t['side'] == 'buy'])
    total_sells = len([t for t in trades_executed if t['side'] == 'sell'])
    total_value = sum(t['value'] for t in trades_executed)
    
    logger.info(f"Execution complete: {total_buys} buys, {total_sells} sells, ${total_value:,.2f} traded")


if __name__ == '__main__':
    execute_signals()
