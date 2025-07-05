import unittest
from src.backtest import engine
from unittest.mock import patch

class TestBacktest(unittest.TestCase):

    @patch('src.ingest.finnhub_client.get_stock_candles')
    @patch('src.train.train.train_model')
    @patch('src.signal.generate.generate_signals')
    @patch('src.exec.execute.execute_signals')
    def test_backtest_smoke_test(self, mock_execute, mock_generate, mock_train, mock_get_candles):
        """
        Smoke test for the backtesting module.
        """
        # Run backtest for a single day
        engine.run_backtest('2023-01-01', '2023-01-01')

        # Verify that the main functions were called
        self.assertTrue(mock_get_candles.called)
        self.assertTrue(mock_train.called)
        self.assertTrue(mock_generate.called)
        self.assertTrue(mock_execute.called)

if __name__ == '__main__':
    unittest.main()
