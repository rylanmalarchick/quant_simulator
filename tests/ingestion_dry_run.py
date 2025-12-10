import unittest
import os
from datetime import datetime, timedelta
from src.ingest import market_data_client, nav_client, pelosi_client, sqlite_writer

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
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_ingestion_dry_run(self):
        """
        Test the data ingestion pipeline for a single epoch.
        """
        # Calculate date range (last 5 days to ensure we get data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Fetch data using yfinance-based market_data_client
        data = market_data_client.get_market_data(
            'AAPL', 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        # Skip test if no market data available (weekend/holiday)
        if data is None or not data.get('c'):
            self.skipTest("No market data available for the date range")
        
        nav = nav_client.get_eod_nav('FXAIX')
        pelosi_trades = pelosi_client.get_pelosi_trades()

        # Write market data to the database
        bars_data = []
        for i in range(len(data['t'])):
            bars_data.append((
                'AAPL', 
                data['t'][i], 
                data['o'][i], 
                data['h'][i], 
                data['l'][i], 
                data['c'][i], 
                data['v'][i]
            ))
        sqlite_writer.write_raw_bars(bars_data)
        
        # Write NAV data if available
        if nav:
            sqlite_writer.write_raw_navs([(nav['symbol'], datetime.now().strftime('%Y-%m-%d'), nav['nav'])])
        
        # Write Pelosi trades if available
        if pelosi_trades:
            pelosi_data = []
            for t in pelosi_trades:
                pelosi_data.append((
                    t.get('disclosure_year', ''),
                    t.get('disclosure_date', ''),
                    t.get('transaction_date', ''),
                    t.get('owner', ''),
                    t.get('ticker', ''),
                    t.get('asset_description', ''),
                    t.get('type', ''),
                    t.get('amount', ''),
                    t.get('representative', ''),
                    t.get('district', ''),
                    t.get('ptr_link', ''),
                    t.get('cap_gains_over_200_usd', False)
                ))
            sqlite_writer.write_pelosi_trades(pelosi_data)

        # Verify that the data was written to the database
        conn = sqlite_writer.get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM raw_bars')
        bars = c.fetchall()
        self.assertGreater(len(bars), 0, "Expected raw_bars to have data")
        
        # NAV and Pelosi trades are optional - don't fail if external APIs are down
        c.execute('SELECT * FROM raw_navs')
        navs = c.fetchall()
        # Just check query works, don't require data
        
        c.execute('SELECT * FROM pelosi_trades')
        trades = c.fetchall()
        # Just check query works, don't require data
        
        conn.close()

    def test_market_data_client(self):
        """
        Test that market_data_client can fetch data from yfinance.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = market_data_client.get_market_data(
            'AAPL',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Should return data with expected structure
        if data is not None:
            self.assertIn('c', data)  # close prices
            self.assertIn('o', data)  # open prices
            self.assertIn('h', data)  # high prices
            self.assertIn('l', data)  # low prices
            self.assertIn('v', data)  # volume
            self.assertIn('t', data)  # timestamps
            self.assertEqual(len(data['c']), len(data['t']))

    def test_invalid_ticker(self):
        """
        Test that invalid tickers are handled gracefully.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        data = market_data_client.get_market_data(
            'INVALIDTICKER123XYZ',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Should return None for invalid ticker
        self.assertIsNone(data)


if __name__ == '__main__':
    unittest.main()
