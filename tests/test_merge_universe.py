import unittest
from unittest.mock import patch
from src.universe import merge

class TestMergeUniverse(unittest.TestCase):

    @patch('src.universe.loader.load_and_validate_universe')
    @patch('src.universe.dynamic.calculate_dynamic_universe')
    @patch('src.ingest.pelosi_client.get_pelosi_trades')
    def test_merge_and_tag_universe(self, mock_get_pelosi_trades, mock_calculate_dynamic_universe, mock_load_and_validate_universe):
        """
        Test merging and tagging the universe.
        """
        # Mock the dependencies to return some dummy data
        mock_load_and_validate_universe.return_value = {
            'funds': ['FXAIX'],
            'quantum': ['IONQ'],
            'dynamic': []
        }
        mock_calculate_dynamic_universe.return_value = ['AAPL']
        mock_get_pelosi_trades.return_value = [{'ticker': 'GOOG'}]
        
        tagged_universe = merge.merge_and_tag_universe()
        self.assertIsNotNone(tagged_universe)
        self.assertIsInstance(tagged_universe, list)
        
        # Check that the symbols are tagged correctly
        for item in tagged_universe:
            if item['symbol'] == 'FXAIX':
                self.assertIn('fund', item['tags'])
            if item['symbol'] == 'IONQ':
                self.assertIn('quantum', item['tags'])
            if item['symbol'] == 'AAPL':
                self.assertIn('dynamic', item['tags'])
            if item['symbol'] == 'GOOG':
                self.assertIn('pelosi', item['tags'])

if __name__ == '__main__':
    unittest.main()
