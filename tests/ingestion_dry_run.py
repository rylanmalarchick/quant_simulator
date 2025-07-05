import unittest
import os
from src.ingest import finnhub_client, nav_client, pelosi_client, sqlite_writer

class TestIngestion(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment.
        """
        # Create a temporary database for testing
        self.db_path = 'test_data.db'
        os.environ['DATABASE_PATH'] = self.db_path
        sqlite_writer.create_tables()

    def tearDown(self):
        """
        Clean up the test environment.
        """
        os.remove(self.db_path)

    def test_ingestion_dry_run(self):
        """
        Test the data ingestion pipeline for a single epoch.
        """
        # Fetch data
        bars = finnhub_client.get_stock_candles('AAPL', 'D', 1577836800, 1577836800)
        nav = nav_client.get_eod_nav('FXAIX')
        pelosi_trades = pelosi_client.get_pelosi_trades()

        # Write data to the database
        sqlite_writer.write_raw_bars([(
            'AAPL', 
            b['t'][0], 
            b['o'][0], 
            b['h'][0], 
            b['l'][0], 
            b['c'][0], 
            b['v'][0]
        ) for b in bars])
        sqlite_writer.write_raw_navs([(nav['symbol'], '2025-07-04', nav['nav'])])
        sqlite_writer.write_pelosi_trades([(
            t['disclosure_year'],
            t['disclosure_date'],
            t['transaction_date'],
            t['owner'],
            t['ticker'],
            t['asset_description'],
            t['type'],
            t['amount'],
            t['representative'],
            t['district'],
            t['ptr_link'],
            t['cap_gains_over_200_usd']
        ) for t in pelosi_trades])

        # Verify that the data was written to the database
        conn = sqlite_writer.get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM raw_bars')
        self.assertGreater(len(c.fetchall()), 0)
        c.execute('SELECT * FROM raw_navs')
        self.assertGreater(len(c.fetchall()), 0)
        c.execute('SELECT * FROM pelosi_trades')
        self.assertGreater(len(c.fetchall()), 0)
        conn.close()

if __name__ == '__main__':
    unittest.main()
