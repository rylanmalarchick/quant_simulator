# CLI Usage

The application provides a command-line interface (CLI) for interacting with the system.

## Commands

### `init`

Initializes the database.

```bash
python src/cli.py init
```

### `ingest`

Ingests data from all sources.

```bash
python src/cli.py ingest
```

### `train-model`

Trains the model.

```bash
python src/cli.py train-model
```

### `signal`

Generates trading signals.

```bash
python src/cli.py signal
```

### `backtest`

Runs a backtest of the trading strategy.

```bash
python src/cli.py backtest --start=YYYY-MM-DD --end=YYYY-MM-DD
```

### `add-symbol`

Adds a symbol to the specified category in `tickers.yml`.

```bash
python src/cli.py add-symbol <SYMBOL> --category=quantum
```

### `remove-symbol`

Removes a symbol from the specified category in `tickers.yml`.

```bash
python src/cli.py remove-symbol <SYMBOL>
```
