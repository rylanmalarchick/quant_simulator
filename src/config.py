import yaml
import os

def load_config():
    """
    Loads the configuration from the .qtick/config.yaml file.
    """
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.qtick', 'config.yaml'))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_tickers():
    """
    Loads the tickers from the tickers.yml file and validates the static lists.
    """
    tickers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tickers.yml'))
    with open(tickers_path, 'r') as f:
        tickers = yaml.safe_load(f)

    # Validate the static lists
    if len(tickers['funds']) != 15:
        raise ValueError("The 'funds' list in tickers.yml must contain 15 tickers.")
    if len(tickers['quantum']) != 13:
        raise ValueError("The 'quantum' list in tickers.yml must contain 13 tickers.")

    return tickers

if __name__ == '__main__':
    config = load_config()
    tickers = load_tickers()
    print("Config loaded successfully:")
    print(config)
    print("\nTickers loaded successfully:")
    print(tickers)
