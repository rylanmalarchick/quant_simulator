import unittest
import os
import pandas as pd
from src.features import builder
from src.ingest import sqlite_writer

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment.
        """
        # Create a temporary database for testing
        self.db_path = 'test_data.db'
        os.environ['DATABASE_PATH'] = self.db_path
        sqlite_writer.create_tables()

        # Create some dummy data
        dummy_bars = [
            ('AAPL', 1672531200, 130.0, 131.0, 129.0, 130.5, 100000),
            ('AAPL', 1672617600, 130.6, 132.0, 130.0, 131.5, 120000),
        ]
        sqlite_writer.write_raw_bars(dummy_bars)

    def tearDown(self):
        """
        Clean up the test environment.
        """
        os.remove(self.db_path)

    def test_feature_engineering_dry_run(self):
        """
        Test the feature engineering pipeline for a single epoch.
        """
        # Build features
        features_df = builder.build_and_persist_features()

        # Verify that the features were created
        self.assertFalse(features_df.empty)

        # Verify that the features were written to the database
        conn = sqlite_writer.get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM features')
        self.assertGreater(len(c.fetchall()), 0)
        conn.close()

if __name__ == '__main__':
    unittest.main()
