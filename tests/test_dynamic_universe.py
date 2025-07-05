import unittest
from unittest.mock import patch
from src.universe import dynamic

class TestDynamicUniverse(unittest.TestCase):

    @patch('src.ingest.finnhub_client.get_stock_candles')
    def test_calculate_dynamic_universe(self, mock_get_stock_candles):
        """
        Test calculating the dynamic universe.
        """
        # Mock the finnhub client to return some dummy data
        mock_get_stock_candles.return_value = {
            'c': [100, 110],
            't': [1672531200, 1675209600]
        }
        
        dynamic_universe = dynamic.calculate_dynamic_universe()
        self.assertIsNotNone(dynamic_universe)
        self.assertIsInstance(dynamic_universe, list)

if __name__ == '__main__':
    unittest.main()
