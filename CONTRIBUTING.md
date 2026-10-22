# Configuration and experiment commands

## Environment

```sh
python -m venv .venv
```

Activate the environment for your shell and install:

```sh
python -m pip install -r requirements.txt
```

Optional training dependencies are in `requirements-ai.txt`. Copy `.env.example` to a local `.env` when using KIS functions. Credentials include application keys, secrets, account/product identifiers, and futures product-code mappings; do not publish their values.

## Non-order paths

```sh
python main.py --dry-run --asset domestic-stock
python main.py --dry-run --asset futures
python main.py --paper-check --asset domestic-stock
python main.py --paper-check --asset futures
```

These paths still depend on the configured data source and network. `--kis-live-probe` enables quote/account probing, not an order submission.

## Simulation and backtesting

```sh
python main.py --paper-sim --asset futures
python backtest_futures.py --watchlist MNQ,MES --period 60d --capital 5000000
python backtest_intraday.py --watchlist 005930,000660 --period 60d --kr-amount 5000000
```

Returned data depends on source coverage. Commission, slippage, exchange hours, contract mapping, and fill assumptions must be considered when interpreting the output.

## Data collection

Inspect a collection plan without downloading market data:

```sh
python scripts/backfill_intraday.py --symbols all --source auto --dry-run
```

Downloaded caches, local reports, databases, token caches, and trained models are not source assets.

## TOA workflow

```sh
python -m toa_ai.cli init-db
python -m toa_ai.cli status
python -m toa_ai.cli smoke-test --epochs 1
```

The separate TOA AI repository contains the package-focused workflow. This repository also includes market-data, broker, and strategy experiments beyond that package.

## Order boundaries

`ENABLE_REAL_ORDER=False` is the default. Futures paper orders require both an explicit enable flag and verified KIS futures endpoints. Endpoint names, transaction IDs, product codes, request fields, and margin behavior must be established against the configured broker environment before enabling an order path.

The commands above do not enable those flags. Synthetic training, backtests, and paper replay cannot establish live execution reliability or future returns.

## Validation

Use module-specific checks for the code being changed. A useful package-level syntax check is:

```sh
python -m compileall toa_ai
```

Do not run a broker probe or order mode merely to validate a documentation change. Keep private accounts and credentials out of logs and reports.
