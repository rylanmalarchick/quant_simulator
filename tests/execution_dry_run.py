import unittest
import os
import pandas as pd
from src.exec import execute
from src.ingest import sqlite_writer
from src.train import train
from unittest.mock import patch

class TestExecution(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment.
        """
        # Create a temporary database for testing
        self.db_path = 'test_data.db'
        os.environ['DATABASE_PATH'] = self.db_path
        sqlite_writer.create_tables()

        # Create some dummy data
        dummy_features = pd.DataFrame({
            'symbol': ['AAPL', 'GOOG'],
            'timestamp': [1672531200, 1672531200],
            'close': [130.5, 2700.0],
            'return_5': [0.1, 0.2],
            'return_15': [0.1, 0.2],
            'return_60': [0.1, 0.2],
            'volatility_20': [0.1, 0.2],
            'volume_ratio_20': [0.1, 0.2],
            'is_pelosi': [0, 1],
            'is_quantum': [1, 0],
            'is_dynamic': [0, 1],
            'category_weight': [0.5, 0.0],
        })
        sqlite_writer.write_features(dummy_features)
        
        # Train a dummy model
        train.train_model()

    def tearDown(self):
        """
        Clean up the test environment.
        """
        os.remove(self.db_path)
        model_path = f"models/model_{pd.Timestamp.now().strftime('%Y%m%d')}.pkl"
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists('signals.log'):
            os.remove('signals.log')

    @patch('src.exec.alpaca_client.submit_order')
    def test_execution_dry_run(self, mock_submit_order):
        """
        Test the execution pipeline for a single epoch.
        """
        # Execute signals
        execute.execute_signals()

        # Verify that the submit_order function was called
        self.assertTrue(mock_submit_order.called)

if __name__ == '__main__':
    unittest.main()
