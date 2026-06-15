# Home PC Codex Handoff

This file is for continuing TOA AI work from the Windows home PC after BTCUSDT
1-minute full training finishes.

## Current Branch

Use this branch:

```powershell
git checkout feature/kis-multi-asset-clients
git pull
```

## Current Goal

Train and evaluate the TOA AI policy model on BTCUSDT Binance 1-minute data.

The Linux laptop already confirmed:

- `toa_ai` package is committed and pushed.
- Windows PowerShell scripts exist under `toa_ai/*.ps1`.
- Binance BTCUSDT 1m ingest produced `1,071,360` bars for 2024-06-01 through 2026-06-14.
- One-month smoke training on 2026-05 completed successfully.
- Full training is expected to be heavy because dataset generation currently builds all samples in memory.

## Windows Setup

From PowerShell:

```powershell
cd C:\auto_trader
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Data Pipeline

Run these from `C:\auto_trader`:

```powershell
.\toa_ai\download_btcusdt_1m.ps1
.\toa_ai\ingest_btcusdt_1m.ps1
.\toa_ai\check_btcusdt_1m.ps1
```

Expected check output:

```text
BTCUSDT 1m rows=1071360 first=2024-06-01T00:00:00+00:00 last=2026-06-14T23:59:00+00:00
```

## Smoke Training

Before full training, run:

```powershell
.\toa_ai\train_btcusdt_smoke.ps1
.\toa_ai\replay_btcusdt_smoke.ps1
```

Smoke training should produce a final JSON with fields like:

- `model_id`
- `model_path`
- `metrics.validation_loss`
- `metrics.action_accuracy`
- `metrics.action_entropy`
- `metrics.no_action_ratio`
- `metrics.prediction_counts`

## Full Training

Run:

```powershell
.\toa_ai\train_btcusdt_full.ps1
```

If editing the script, batch size can be adjusted in:

```powershell
--batch-size 256 `
```

Suggested values:

- 512 for 32GB RAM
- 1024 for 64GB+ RAM

Current trainer may use CPU unless `toa_ai/trainer.py` has been changed to select CUDA. If GPU is active, keep watching RAM as well.

## After Full Training

Run:

```powershell
.\toa_ai\replay_btcusdt_latest.ps1
.\.venv\Scripts\python.exe -m toa_ai.cli --db-path data\toa_ai\btc_1m.sqlite3 status
```

Then promote if the latest model passes the quality gate:

```powershell
.\.venv\Scripts\python.exe -m toa_ai.cli --db-path data\toa_ai\btc_1m.sqlite3 promote --model-id latest
```

## Result Files To Create

Create small text/JSON outputs that can be committed back for review:

```powershell
.\.venv\Scripts\python.exe -m toa_ai.cli --db-path data\toa_ai\btc_1m.sqlite3 status > btc_status.json
.\toa_ai\replay_btcusdt_latest.ps1 > btc_replay_latest.json
```

Commit only small result files and source changes. Do not commit large raw data,
SQLite DBs, or model binaries unless Git LFS is intentionally set up.

```powershell
git add btc_status.json btc_replay_latest.json
git commit -m "add btc training results"
git push
```

## Important Bottleneck

The main full-training bottleneck is in `toa_ai/features.py`.

Current slow path:

- `build_policy_dataset_for_symbol`
- `_entry_label`
- `_holding_label`
- `future.iterrows()`

Why it is slow:

- Full BTC data has about 1.07M bars.
- The dataset builder creates roughly 2.14M samples.
- Each sample checks up to 30 future bars.
- Pandas `iterrows()` creates a Series per row and is very slow.

Recommended next optimization:

1. Keep `build_feature_frame` mostly unchanged.
2. In `build_policy_dataset_for_symbol`, precompute:
   - `high_values = data["high"].to_numpy(dtype=np.float64)`
   - `low_values = data["low"].to_numpy(dtype=np.float64)`
   - `close_values_label = data["close"].to_numpy(dtype=np.float64)`
3. Rewrite `_entry_label` and `_holding_label` to use numpy slices instead of pandas `iterrows()`.
4. Preserve label semantics exactly:
   - Entry label: if stop and target hit in the same future bar, choose `NO_ACTION` with stop return.
   - Holding label priority per bar: stop first, then target2, then target.
5. Add an equivalence test comparing old and new label functions on random and real BTC data.

Do not use `np.argmax(mask)` without checking `mask.any()` first, because all-False masks return index 0.

## Second Bottleneck

Even after removing `iterrows()`, full training still stores the whole dataset in memory:

```python
xs.append(base_matrix[start : end + 1].copy())
...
np.asarray(xs, dtype=np.float32)
```

For 2 years of 1-minute BTC data, this can use tens of GB of RAM.

If full training remains too heavy, the next design change should be one of:

- streaming PyTorch `Dataset`
- memmap-backed feature windows
- precomputed labels plus on-demand window slicing
- chunked training by time range

## What To Report Back

After training/replay, send back:

- full training final JSON
- replay latest JSON
- `btc_status.json`
- whether GPU was used
- peak RAM usage
- total training time

## Safety Notes

- Do not run live or paper order commands from this handoff.
- This work is model training and replay only.
- Profit metrics are diagnostic only at this stage.
