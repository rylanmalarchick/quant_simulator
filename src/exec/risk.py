from src.config import load_config

def check_risk_controls(order_size, daily_turnover):
    """
    Checks if the order is within the defined risk controls.
    """
    config = load_config()
    
    max_position_size = config['execution']['max_position_size']
    max_daily_turnover = config['execution']['max_daily_turnover']
    
    if order_size > max_position_size:
        print(f"Order size ({order_size}) exceeds max position size ({max_position_size}).")
        return False
        
    if daily_turnover > max_daily_turnover:
        print(f"Daily turnover ({daily_turnover}) exceeds max daily turnover ({max_daily_turnover}).")
        return False
        
    return True

if __name__ == '__main__':
    # Example usage
    if check_risk_controls(0.06, 0.1):
        print("Order is within risk limits.")
    else:
        print("Order exceeds risk limits.")
        
    if check_risk_controls(0.04, 0.3):
        print("Order is within risk limits.")
    else:
        print("Order exceeds risk limits.")
