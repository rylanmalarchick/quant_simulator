import unittest
import os
import yaml
from src import config

class TestConfig(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment.
        """
        # Create dummy config files
        self.config_dir = '.qtick'
        self.config_path = os.path.join(self.config_dir, 'config.yaml')
        self.tickers_path = 'tickers.yml'

        os.makedirs(self.config_dir, exist_ok=True)

        with open(self.config_path, 'w') as f:
            yaml.dump({'api_keys': {'finnhub': 'test_key'}}, f)

        with open(self.tickers_path, 'w') as f:
            yaml.dump({
                'funds': ['FXAIX'] * 15,
                'quantum': ['IONQ'] * 13,
                'dynamic': []
            }, f)

    def tearDown(self):
        """
        Clean up the test environment.
        """
        os.remove(self.config_path)
        os.rmdir(self.config_dir)
        os.remove(self.tickers_path)

    def test_load_config(self):
        """
        Test loading the config.yaml file.
        """
        loaded_config = config.load_config()
        self.assertEqual(loaded_config['api_keys']['finnhub'], 'test_key')

    def test_load_tickers(self):
        """
        Test loading the tickers.yml file.
        """
        loaded_tickers = config.load_tickers()
        self.assertEqual(len(loaded_tickers['funds']), 15)
        self.assertEqual(len(loaded_tickers['quantum']), 13)

if __name__ == '__main__':
    unittest.main()
