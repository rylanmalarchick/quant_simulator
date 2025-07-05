import unittest
import os
import pandas as pd
from src.signal import generate
from src.ingest import sqlite_writer
from src.train import train

class TestSignalGeneration(unittest.TestCase):

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
            'symbol': ['AAPL'] * 20,
            'timestamp': range(20),
            'close': range(20),
            'return_5': [0.1] * 20,
            'return_15': [0.1] * 20,
            'return_60': [0.1] * 20,
            'volatility_20': [0.1] * 20,
            'volume_ratio_20': [0.1] * 20,
            'is_pelosi': [0] * 20,
            'is_quantum': [0] * 20,
            'is_dynamic': [0] * 20,
            'category_weight': [0.0] * 20,
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

    def test_signal_generation_dry_run(self):
        """
        Test the signal generation pipeline for a single epoch.
        """
        # Generate signals
        generate.generate_signals()

        # Verify that the signals were logged
        self.assertTrue(os.path.exists('signals.log'))

if __name__ == '__main__':
    unittest.main()
