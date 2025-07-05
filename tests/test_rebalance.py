import unittest
from unittest.mock import patch
from src.universe import rebalance

class TestRebalance(unittest.TestCase):

    @patch('src.ingest.finnhub_client.get_stock_candles')
    @patch('src.universe.loader.load_and_validate_universe')
    def test_calculate_rebalance_weights(self, mock_load_and_validate_universe, mock_get_stock_candles):
        """
        Test calculating the rebalance weights.
        """
        # Mock the dependencies to return some dummy data
        mock_load_and_validate_universe.return_value = {
            'funds': ['FXAIX'],
            'quantum': ['IONQ'],
            'dynamic': []
        }
        mock_get_stock_candles.return_value = {
            'c': [100, 110],
            't': [1672531200, 1675209600]
        }
        
        rebalance_weights = rebalance.calculate_rebalance_weights()
        self.assertIsNotNone(rebalance_weights)
        self.assertIsInstance(rebalance_weights, dict)
        self.assertIn('funds', rebalance_weights)
        self.assertIn('quantum', rebalance_weights)

if __name__ == '__main__':
    unittest.main()
