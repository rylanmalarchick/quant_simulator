from src.universe.loader import load_and_validate_universe
from src.universe.dynamic import calculate_dynamic_universe
from src.ingest.pelosi_client import get_pelosi_trades

def merge_and_tag_universe():
    """
    Merges the different universes and tags each symbol with its category.
    """
    universe = load_and_validate_universe()
    dynamic_universe = calculate_dynamic_universe()
    pelosi_trades = get_pelosi_trades()
    
    # Get Pelosi tickers
    pelosi_tickers = {trade['ticker'] for trade in pelosi_trades}
    
    # Merge universes
    merged_universe = set(universe['funds']) | set(universe['quantum']) | set(dynamic_universe) | pelosi_tickers
    
    # Tag symbols
    tagged_universe = []
    for symbol in merged_universe:
        tags = []
        if symbol in universe['funds']:
            tags.append('fund')
        if symbol in universe['quantum']:
            tags.append('quantum')
        if symbol in dynamic_universe:
            tags.append('dynamic')
        if symbol in pelosi_tickers:
            tags.append('pelosi')
        tagged_universe.append({'symbol': symbol, 'tags': tags})
        
    return tagged_universe

if __name__ == '__main__':
    tagged_universe = merge_and_tag_universe()
    print("Tagged universe:")
    print(tagged_universe)
