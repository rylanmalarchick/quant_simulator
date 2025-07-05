# Configuration

The application is configured using two YAML files: `.qtick/config.yaml` and `tickers.yml`.

## `.qtick/config.yaml`

This file contains the main configuration for the application, including API keys, data provider settings, model parameters, and execution settings.

```yaml
api_keys:
  finnhub: YOUR_FINNHUB_KEY
  alpaca_key: YOUR_ALPACA_KEY
  alpaca_secret: YOUR_ALPACA_SECRET

data_provider: finnhub
bar_resolution: "15min"
universe_file: tickers.yml
top_k: 5

model:
  type: lightgbm
  params:
    max_depth: 4
    learning_rate: 0.1
    num_leaves: 31

execution:
  broker: alpaca
  max_position_size: 0.05
  max_daily_turnover: 0.2
```

## `tickers.yml`

This file defines the universe of stocks to be traded. It is divided into three categories:

- `funds`: A static list of mutual funds.
- `quantum`: A static list of quantum computing stocks.
- `dynamic`: A list of top-performing stocks that is automatically populated.

```yaml
funds:
  - FXAIX
  - FUSVX
  - FSRBX
  - FDEQX
  - FSHOX
  - FSPSX
  - FSELX
  - FSCSX
  - FUSEX
  - FUSVX
  - FSPTX
  - FSDAX
  - FSPCX
  - FNCMX
  - FBMPX

quantum:
  - IONQ
  - IBM
  - NVDA
  - AMD
  - MSFT
  - GOOG
  - AMZN
  - HON
  - QRVO
  - LSCC
  - MRVL
  - KLIC
  - QBTS

dynamic: []
```
