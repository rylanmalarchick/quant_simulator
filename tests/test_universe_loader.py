import unittest
import os
import yaml
from src.universe import loader

class TestUniverseLoader(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment.
        """
        # Create a dummy tickers.yml file
        self.tickers_path = 'tickers.yml'
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
        os.remove(self.tickers_path)

    def test_load_and_validate_universe(self):
        """
        Test loading and validating the universe.
        """
        universe = loader.load_and_validate_universe()
        self.assertIsNotNone(universe)

    def test_invalid_dynamic_universe(self):
        """
        Test that an error is raised for an invalid dynamic universe.
        """
        with open(self.tickers_path, 'w') as f:
            yaml.dump({
                'funds': ['FXAIX'] * 15,
                'quantum': ['IONQ'] * 13,
                'dynamic': ['AAPL'] * 41
            }, f)
        with self.assertRaises(ValueError):
            loader.load_and_validate_universe()

    def test_invalid_total_universe(self):
        """
        Test that an error is raised for an invalid total universe.
        """
        with open(self.tickers_path, 'w') as f:
            yaml.dump({
                'funds': ['FXAIX'] * 15,
                'quantum': ['IONQ'] * 13,
                'dynamic': ['AAPL'] * 50
            }, f)
        with self.assertRaises(ValueError):
            loader.load_and_validate_universe()

if __name__ == '__main__':
    unittest.main()
