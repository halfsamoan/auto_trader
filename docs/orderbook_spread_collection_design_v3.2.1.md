# V3.2.1 Orderbook / Spread Collection Design

## Scope

This version does not implement live orderbook collection. It only documents the
data contract needed by later audit and liquidity gates.

No order APIs are used by this design.

## Required Fields

Each orderbook observation should be stored with these fields:

- `timestamp`: Asia/Seoul timestamp for the observation.
- `code`: six-digit domestic-stock code.
- `best_bid`: best bid price.
- `best_ask`: best ask price.
- `bid_size_1`: best bid quantity.
- `ask_size_1`: best ask quantity.
- `spread`: `best_ask - best_bid`.
- `spread_pct`: `spread / midpoint`, where midpoint is `(best_bid + best_ask) / 2`.
- `source`: `kis_websocket_orderbook` or another explicit quote-only source.
- `collected_at`: local collection timestamp.

Optional depth fields can be added later as `bid_price_2`, `ask_price_2`,
`bid_size_2`, `ask_size_2`, and so on.

## Storage

Use the existing ignored cache directory:

```text
data/orderbook_cache/{code}_orderbook.csv
```

Append idempotently by `(timestamp, code)` and keep the last observation if the
same timestamp is collected twice.

## Collection Path

Preferred source is KIS WebSocket quotation/orderbook data. REST polling can be
used only if KIS explicitly provides a quote-only endpoint with safe rate limits.

Collector requirements:

- Reuse existing token/session handling.
- Do not print app keys, secrets, tokens, or account identifiers.
- Add per-symbol throttling and reconnect backoff.
- Continue to the next symbol after repeated failures.
- Emit `order_api_called=False` in summaries.

## Audit Integration

Until this cache exists, `missing_orderbook_count` and
`spread_missing_ratio=1.0` are expected. Trading gates should treat missing
orderbook data as a liquidity block for shadow/paper-watch decisions, not as a
reason to call any order API.
