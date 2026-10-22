# Auto Trader

A Python research workspace for domestic-stock and overseas-futures backtests, market data, and KIS paper-trading integration.

## Implementation

- **Broker adapters:** separates shared KIS behavior from domestic-stock and futures clients.
- **Market data:** collects intraday data into local caches for signal analysis and backtesting.
- **Strategy execution:** combines signal generation, position handling, and risk checks with explicit execution modes.
- **Model experiments:** includes policy-training and replay utilities, including the separately published [TOA AI package](https://github.com/halfsamoan/TOa_ai).

[Broker clients](clients/) · [Execution modes](main.py) · [Risk manager](risk_manager.py)

## Run without orders

From the repository root in a Python environment:

```sh
python -m pip install -r requirements.txt
python main.py --dry-run --asset domestic-stock
```

The dry run does not submit orders. Market-data access requires a network connection. KIS-specific paths require credentials configured locally from `.env.example`.

## Execution boundaries

Live orders are disabled by `ENABLE_REAL_ORDER=False`. KIS futures paper orders are also blocked until their verification and enable flags are set. `--kis-live-probe` is a quote/account probe, not an order command.

Broker endpoint verification remains incomplete. Backtest and paper-simulation results do not establish live execution reliability or future returns.

[Configuration and experiment commands](CONTRIBUTING.md)

## License

[MIT](LICENSE).
