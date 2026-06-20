# auto_trader

`auto_trader` is a Python research and paper-trading lab for Korea domestic stocks, CME micro futures experiments, and TOA AI policy testing. It is built around conservative defaults: dry-run and paper modes are the normal path, real order execution is blocked unless several explicit safety gates are changed and verified.

This repository is public for study, simulation, and extension. It is not financial advice, and it should not be used with real money until the broker routes, risk controls, and market assumptions have been independently audited.

## What is included

- KIS Open API client scaffolding for domestic-stock quotation, balance, and guarded paper-order flows.
- Overseas futures paper-simulation and KIS futures probe scaffolding for `MNQ` and `MES`.
- Intraday backtests, paper simulations, risk management, and behavior-tree experiments.
- PatchTST-style AI signal/policy training utilities and evaluation scripts.
- TOA AI memory, training, promotion, replay, and paper-once CLI workflows.

## Safety status

- `ENABLE_REAL_ORDER` is `False` by default.
- Futures KIS paper orders are blocked by default with `ENABLE_FUTURES_KIS_PAPER_ORDER=False` and `FUTURES_KIS_ENDPOINTS_VERIFIED=False`.
- `--kis-live-probe` calls quote/account probe endpoints only; it does not place orders.
- `.env`, token caches, runtime paper-trade files, model artifacts, local databases, and market-data caches are ignored by Git.
- Account numbers are masked in futures probe output where the code prints account context.

## Requirements

- Python 3.11 or newer is recommended.
- KIS Open API credentials are only needed for KIS quotation, account, paper-check, or broker-probe paths.
- Optional AI workflows require the packages in `requirements-ai.txt`.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional AI dependencies:

```bash
python -m pip install -r requirements-ai.txt
```

## Configuration

Copy the sample environment file and fill in your own values locally:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`, API keys, app secrets, access tokens, account numbers, downloaded market data, local databases, or trained model artifacts.

Important KIS variables:

- `KIS_MODE=paper`
- `KIS_APP_KEY`
- `KIS_APP_SECRET`
- `KIS_ACCOUNT_NO`
- `KIS_ACCOUNT_PRODUCT_CODE`
- `KIS_FUTURES_ACCOUNT_NO`
- `KIS_FUTURES_ACCOUNT_PRODUCT_CODE`
- `KIS_FUTURES_MNQ_PRODUCT_CODE`
- `KIS_FUTURES_MES_PRODUCT_CODE`

## Common commands

Safe dry runs:

```bash
python main.py --dry-run --asset domestic-stock
python main.py --dry-run --asset futures
```

Paper-check paths:

```bash
python main.py --paper-check --asset domestic-stock
python main.py --paper-check --asset futures
python main.py --paper-check --asset futures --kis-live-probe
```

Paper simulation without broker order calls:

```bash
python main.py --paper-sim --asset futures
python backtest_futures.py --watchlist MNQ,MES --period 60d --capital 5000000
python backtest_intraday.py --watchlist 005930,000660 --period 60d --kr-amount 5000000
```

Domestic-stock paper watch is guarded and should only be used after reviewing the code, environment, and account setup:

```bash
python main.py --paper-watch --asset domestic-stock --interval-sec 300
```

Paper order paths require explicit flags and paper-mode configuration. Review the code and broker documentation before enabling them:

```bash
python main.py --paper --asset domestic-stock --allow-paper-order
python main.py --paper --asset futures --allow-paper-order
```

## Data collection

Dry-run the domestic intraday collection plan:

```bash
python scripts/backfill_intraday.py --symbols all --source auto --dry-run
```

Collect quote-only data into ignored local caches:

```bash
python scripts/backfill_intraday.py --symbols 005930,000660 --source yfinance --period 60d
python scripts/collect_intraday_data.py --universe ai_train --source auto --period 5d --interval 5m
```

Generated files under `data/intraday_cache`, `data/external`, `data/toa_ai`, `reports`, and `ai/model_store` are intentionally local-only.

## TOA AI workflow

Initialize or inspect the TOA memory database:

```bash
python -m toa_ai.cli init-db
python -m toa_ai.cli status
```

Run a synthetic smoke test:

```bash
python -m toa_ai.cli smoke-test --epochs 1
```

Ingest local cache data, train, promote, and replay:

```bash
python -m toa_ai.cli ingest-cache --cache-dir data/intraday_cache --timeframe 5m --limit-symbols 5
python -m toa_ai.cli train --epochs 4 --max-symbols 5
python -m toa_ai.cli promote --model-id latest
python -m toa_ai.cli replay --model-id champion --max-symbols 3
```

## Notes and cautions

- Backtest, paper-sim, and smoke-test results do not guarantee future performance.
- yfinance intraday data has limited history and may differ from broker data.
- KIS overseas futures endpoint names, TR IDs, body fields, product-code mappings, and margin logic must be manually verified before any order path is enabled.
- The project currently favors experimentation and safety gates over production-grade reliability.
- Keep credentials local. If a secret is ever committed, rotate it immediately and rewrite the Git history before relying on the repository.
- Check tax, commission, slippage, exchange holidays, trading halts, daily price limits, and broker-specific order semantics before expanding execution scope.

## Validation

Recommended checks before publishing or merging changes:

```bash
python -m compileall .
python -m toa_ai.cli smoke-test --epochs 1
python scripts/backfill_intraday.py --symbols 005930 --source yfinance --dry-run
```

## Roadmap

- Verify KIS overseas futures paper-trading endpoints, TR IDs, request bodies, and error handling against official broker samples.
- Add automated tests for risk gates, order blocking, token redaction, paper-sim fills, and TOA promotion logic.
- Add CI for linting, import checks, smoke tests, and secret scanning.
- Separate broker adapters from strategy logic so simulations and live probes share less mutable state.
- Improve model registry metadata and reproducibility for AI training runs.
- Add documented examples for custom universes, external CSV ingestion, and replay reports.
- Harden logging so all broker/account/token details are centrally redacted.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
