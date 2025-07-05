from src.config import load_tickers

def load_and_validate_universe():
    """
    Loads the universe from tickers.yml and validates it.
    """
    tickers = load_tickers()
    
    # Enforce validation rules
    if len(tickers['dynamic']) > 40:
        raise ValueError("The 'dynamic' list in tickers.yml cannot contain more than 40 tickers.")
    
    total_tickers = len(tickers['funds']) + len(tickers['quantum']) + len(tickers['dynamic'])
    if total_tickers > 75:
        raise ValueError(f"The total number of tickers cannot exceed 75. Currently: {total_tickers}")
        
    return tickers

if __name__ == '__main__':
    universe = load_and_validate_universe()
    print("Universe loaded and validated successfully:")
    print(universe)
