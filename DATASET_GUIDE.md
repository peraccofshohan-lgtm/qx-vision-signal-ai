# Dataset guide

## Required OHLC schema

```csv
timestamp,open,high,low,close,volume,asset,timeframe
2025-01-01T00:00:00Z,1.1000,1.1010,1.0990,1.1006,1234,EURUSD,1m
```

`volume` may be blank. `asset` and `timeframe` are required so windows never cross instruments or timeframes. High/low invariants are checked.

## Label definition

For a feature window ending at prediction time `t`, the target is the **next completed row** at `t+1`: `UP` when `close > open`, `DOWN` when `close < open`; doji targets are skipped by the candidate trainer rather than forced into a direction. The current running screenshot candle is never treated as the target.

## Quality and leakage controls

1. Sort per asset and timestamp.
2. Deduplicate windows using a stable fingerprint.
3. Split chronologically into train, validation, and holdout.
4. Purge at least one row at boundaries; use larger gaps for longer lookbacks.
5. Fit normalization, OOD bounds, and model weights on train only.
6. Fit calibration on validation only.
7. Keep holdout untouched until final evaluation.
8. Reject feature names containing future/target fields and timestamps later than prediction time.
9. Keep real and synthetic screenshot sets separate.

## Candidate dataset lifecycle

New outcome labels are stored separately from immutable prediction-time values. They enter `candidate_dataset` only after validation, hash deduplication, and quality filtering. A candidate is trained and evaluated against the champion; it is promoted only when accuracy, selective accuracy, calibration, coverage, regime stability, and worst-regime gates pass. There is no instant online weight update.

## Screenshot simulation

`qxvision.synthetic.render` renders known OHLC windows into deterministic chart-like PPM frames with varied colors, backgrounds, grid, spacing, and sizes. These images test candle count, direction, body, wick, and chronological order. They are not real market samples and are not a substitute for a real holdout.

## Privacy

Do not put private screenshots, credentials, or proprietary datasets in Git. Keep datasets outside the repository or in an ignored path. The Android app stores a hash and metadata by default, not an indefinite screenshot archive.
