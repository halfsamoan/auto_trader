# Futures Same-Asset Validation Plan

Generated: 2026-06-14

## Scope

PHASE B domestic-stock results cannot confirm an overseas futures edge. MNQ/MES
paper-sim must remain simulation-only until the same asset, same session, same
cost unit, and same contract metadata are validated with local data.

## Current Gate

- `ENABLE_FUTURES_KIS_PAPER_ORDER` must remain `False`.
- `FUTURES_KIS_ENDPOINTS_VERIFIED` must remain `False`.
- Contract metadata `is_kis_paper_order_enabled` must remain `False`.
- No futures KIS order endpoint can be called from this validation plan.
- Missing local `MNQ`/`MES` intraday cache means status is
  `EDGE_CANDIDATE_REQUIRES_SAME_ASSET_VALIDATION`.

## Required Data

- Local 5-minute OHLCV cache for `MNQ` and `MES`, or explicit equivalent
  yfinance continuous symbols mapped to contract metadata.
- Contract roll handling note for each sample window.
- Tick size, tick value, round-trip commission, and slippage assumptions in USD.
- Session calendar and timezone normalization.

## Validation Steps

1. Build same-asset feature matrices for MNQ/MES only.
2. Rebuild triple-barrier labels using futures-specific tick and notional costs.
3. Run geometry sweep in label-only mode, then train-each mode only if sample
   counts pass the same train/valid/test readiness thresholds.
4. Run native CPCV with `AI_PURGE_GAP_BARS=96` or a stricter futures-specific
   purge derived from the holding horizon.
5. Run meta-labeling with costs expressed in the same return units as labels.
6. Require positive CPCV net EV, acceptable DSR/PBO, and positive one-time test
   net EV before calling the result a candidate.

## Rejection Conditions

- Any missing or mixed asset cache for the evaluated futures symbol.
- Any use of domestic-stock cost assumptions as futures return costs without a
  documented notional conversion.
- Any path that enables KIS futures paper orders.
- Any result that depends on test-set threshold tuning.

## Output Status

Until the above steps pass, futures status remains:

`EDGE_CANDIDATE_REQUIRES_SAME_ASSET_VALIDATION`
