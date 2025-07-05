import unittest
from click.testing import CliRunner
from src import cli

class TestCli(unittest.TestCase):

    def test_init_command(self):
        """
        Test the init command.
        """
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['init'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Database initialized.", result.output)

    def test_ingest_command(self):
        """
        Test the ingest command.
        """
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['ingest'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Ingesting data...", result.output)

    def test_train_model_command(self):
        """
        Test the train command.
        """
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['train-model'])
        self.assertEqual(result.exit_code, 0)

    def test_signal_command(self):
        """
        Test the signal command.
        """
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['signal'])
        self.assertEqual(result.exit_code, 0)

    def test_add_symbol_command(self):
        """
        Test the add-symbol command.
        """
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['add-symbol', 'TEST', '--category', 'quantum'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Added TEST to quantum.", result.output)

    def test_remove_symbol_command(self):
        """
        Test the remove-symbol command.
        """
        runner = CliRunner()
        runner.invoke(cli.cli, ['add-symbol', 'TEST', '--category', 'quantum'])
        result = runner.invoke(cli.cli, ['remove-symbol', 'TEST', '--category', 'quantum'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Removed TEST from quantum.", result.output)

if __name__ == '__main__':
    unittest.main()
