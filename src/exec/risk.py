"""
Risk management module for quantSim.

Provides risk controls and position sizing calculations.
Includes crisis-aware position sizing that reduces exposure during market stress.
"""

from src.config import load_config
from src.logging import get_logger

logger = get_logger(__name__)

# Crisis score thresholds for position scaling
CRISIS_THRESHOLDS = {
    'normal': 30,      # Score < 30: Full position sizes
    'elevated': 50,    # Score 30-50: Reduce to 75% of normal
    'high': 70,        # Score 50-70: Reduce to 50% of normal
    'critical': 100,   # Score > 70: Reduce to 25% of normal (or go to cash)
}


def get_crisis_multiplier(crisis_score: float = None) -> float:
    """
    Get position size multiplier based on crisis score.
    
    This implements the stat-mech inspired risk reduction:
    - When markets show crisis signatures, reduce position sizes
    - Allows staying in the market but with reduced risk
    
    Args:
        crisis_score: Crisis score from 0-100 (None = don't adjust)
        
    Returns:
        Multiplier for position sizes (0.25 to 1.0)
    """
    if crisis_score is None:
        return 1.0
    
    if crisis_score < CRISIS_THRESHOLDS['normal']:
        multiplier = 1.0
        level = 'NORMAL'
    elif crisis_score < CRISIS_THRESHOLDS['elevated']:
        multiplier = 0.75
        level = 'ELEVATED'
    elif crisis_score < CRISIS_THRESHOLDS['high']:
        multiplier = 0.50
        level = 'HIGH'
    else:
        multiplier = 0.25
        level = 'CRITICAL'
    
    logger.info(f"Crisis score: {crisis_score:.1f} ({level}) -> Position multiplier: {multiplier:.0%}")
    return multiplier


def get_current_crisis_score() -> float:
    """
    Fetch the current crisis score from the detector.
    
    Returns:
        Crisis score (0-100), or None if unavailable
    """
    try:
        from src.crisis.detector import CrisisDetector
        
        detector = CrisisDetector(lookback_days=252)
        detector.calculate_all_indicators()
        score = detector.calculate_composite_score()
        
        logger.info(f"Current crisis score: {score:.1f}/100")
        return score
    except Exception as e:
        logger.warning(f"Could not fetch crisis score: {e}")
        return None


def check_risk_controls(order_size: float, daily_turnover: float) -> bool:
    """
    Checks if the order is within the defined risk controls.
    
    Args:
        order_size: Order size as a fraction of account equity (0.0 to 1.0)
        daily_turnover: Daily turnover as a fraction of account equity
        
    Returns:
        True if order passes risk controls, False otherwise
    """
    config = load_config()
    
    max_position_size = config['execution'].get('max_position_size', 0.10)
    max_daily_turnover = config['execution'].get('max_daily_turnover', 0.25)
    
    if order_size > max_position_size:
        logger.warning(f"Order size ({order_size:.2%}) exceeds max position size ({max_position_size:.2%}).")
        return False
        
    if daily_turnover > max_daily_turnover:
        logger.warning(f"Daily turnover ({daily_turnover:.2%}) exceeds max daily turnover ({max_daily_turnover:.2%}).")
        return False
        
    return True


def calculate_position_size(
    account_equity: float,
    price: float,
    risk_per_trade: float = 0.02,
    max_position_pct: float = 0.10,
    stop_loss_pct: float = 0.05,
    crisis_score: float = None,
    use_crisis_scaling: bool = True
) -> int:
    """
    Calculate the number of shares to buy based on risk management rules.
    
    Uses the following approach:
    1. Calculate max $ risk = account_equity * risk_per_trade
    2. Calculate risk per share = price * stop_loss_pct
    3. Calculate shares by risk = max $ risk / risk per share
    4. Calculate shares by max position = (account_equity * max_position_pct) / price
    5. Apply crisis multiplier (reduces size during market stress)
    6. Return the minimum of the two (most conservative)
    
    Args:
        account_equity: Total account value
        price: Current price of the stock
        risk_per_trade: Maximum risk per trade as fraction of account (default 2%)
        max_position_pct: Maximum position size as fraction of account (default 10%)
        stop_loss_pct: Stop loss distance as fraction of entry price (default 5%)
        crisis_score: Optional crisis score (0-100) for position scaling
        use_crisis_scaling: Whether to apply crisis-based position reduction
        
    Returns:
        Number of shares to buy (integer)
    """
    if price <= 0 or account_equity <= 0:
        return 0
    
    # Get crisis multiplier if enabled
    if use_crisis_scaling:
        if crisis_score is None:
            # Try to fetch current crisis score (cached for performance)
            crisis_score = get_current_crisis_score()
        crisis_multiplier = get_crisis_multiplier(crisis_score)
    else:
        crisis_multiplier = 1.0
    
    # Method 1: Risk-based sizing
    # How much $ can we risk on this trade?
    max_risk_dollars = account_equity * risk_per_trade
    
    # How much do we lose per share if stop is hit?
    risk_per_share = price * stop_loss_pct
    
    # How many shares can we buy with that risk budget?
    if risk_per_share > 0:
        shares_by_risk = int(max_risk_dollars / risk_per_share)
    else:
        shares_by_risk = 0
    
    # Method 2: Position size limit
    # Maximum $ we can put in a single position
    max_position_dollars = account_equity * max_position_pct
    shares_by_position = int(max_position_dollars / price)
    
    # Take the more conservative (smaller) of the two
    shares = min(shares_by_risk, shares_by_position)
    
    # Apply crisis multiplier (reduces position during market stress)
    shares = int(shares * crisis_multiplier)
    
    # Ensure at least 0 shares (never negative)
    return max(0, shares)


def calculate_stop_loss(entry_price: float, side: str, stop_pct: float = 0.05) -> float:
    """
    Calculate stop loss price.
    
    Args:
        entry_price: Entry price
        side: 'long' or 'short'
        stop_pct: Stop loss percentage (default 5%)
        
    Returns:
        Stop loss price
    """
    if side == 'long':
        return entry_price * (1 - stop_pct)
    else:  # short
        return entry_price * (1 + stop_pct)


def calculate_take_profit(entry_price: float, side: str, target_pct: float = 0.10) -> float:
    """
    Calculate take profit price.
    
    Args:
        entry_price: Entry price
        side: 'long' or 'short'
        target_pct: Profit target percentage (default 10%)
        
    Returns:
        Take profit price
    """
    if side == 'long':
        return entry_price * (1 + target_pct)
    else:  # short
        return entry_price * (1 - target_pct)


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    side: str
) -> float:
    """
    Calculate risk/reward ratio for a trade.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        side: 'long' or 'short'
        
    Returns:
        Risk/reward ratio (higher is better)
    """
    if side == 'long':
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:  # short
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    
    if risk <= 0:
        return 0.0
    
    return reward / risk


if __name__ == '__main__':
    # Example usage
    account = 100000
    price = 150
    
    print("=" * 50)
    print("CRISIS-AWARE POSITION SIZING DEMO")
    print("=" * 50)
    
    # Normal conditions
    print("\n--- NORMAL CONDITIONS (score=20) ---")
    shares = calculate_position_size(account, price, crisis_score=20)
    print(f"Position size: {shares} shares (${shares * price:,})")
    print(f"Position %: {(shares * price / account):.1%}")
    
    # Elevated stress
    print("\n--- ELEVATED STRESS (score=45) ---")
    shares = calculate_position_size(account, price, crisis_score=45)
    print(f"Position size: {shares} shares (${shares * price:,})")
    print(f"Position %: {(shares * price / account):.1%}")
    
    # High stress
    print("\n--- HIGH STRESS (score=60) ---")
    shares = calculate_position_size(account, price, crisis_score=60)
    print(f"Position size: {shares} shares (${shares * price:,})")
    print(f"Position %: {(shares * price / account):.1%}")
    
    # Critical
    print("\n--- CRITICAL (score=80) ---")
    shares = calculate_position_size(account, price, crisis_score=80)
    print(f"Position size: {shares} shares (${shares * price:,})")
    print(f"Position %: {(shares * price / account):.1%}")
    
    # Stop/target calculations
    print("\n--- STOP LOSS / TAKE PROFIT ---")
    stop = calculate_stop_loss(price, 'long')
    target = calculate_take_profit(price, 'long')
    rr = calculate_risk_reward_ratio(price, stop, target, 'long')
    
    print(f"Entry: ${price:.2f}")
    print(f"Stop loss: ${stop:.2f}")
    print(f"Take profit: ${target:.2f}")
    print(f"Risk/Reward: {rr:.2f}")
