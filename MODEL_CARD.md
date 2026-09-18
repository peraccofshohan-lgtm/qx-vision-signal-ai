# Model card — QX Vision Signal AI V1

## Status

**UNTRAINED / NO PRODUCTION ARTIFACT.** No market dataset or model weights are bundled. The Android app will not display a model-derived probability in this state.

## Purpose

Research and local analysis of the direction of the next completed one-minute candle from a user-provided chart screenshot. It is not financial advice and does not guarantee accuracy or profit.

## Input and output

Input is a screenshot. The vision layer returns normalized candle geometry, a quality report, and a partial-observation flag for the right-most candle. The model target is future-candle direction (`UP` or `DOWN`) only. The deployed policy may also return `NO_TRADE`.

## Training methodology

User-supplied OHLC data is sorted by asset and timestamp. Lookback windows are chronological, with a purge gap between train, validation, and holdout. Feature timestamps cannot exceed prediction timestamps, and target/future columns are rejected. Train-only robust bounds support OOD detection. Logistic regression is a reproducible baseline; candidate calibration uses validation only.

## Evaluation

The evaluator reports accepted predictions, coverage, selective accuracy, Brier score, log loss, ECE, confidence buckets, regime breakdown, and streaks. Walk-forward split helpers are provided. The candidate must beat a previous-candle baseline on the later holdout before it can be written as a registry artifact. No metrics are published for this checkout because no data was supplied.

## Calibration and abstention

Platt scaling is selected only from validation data. The decision policy rejects low quality, too few candles, extreme volatility, ambiguous structure, OOD inputs, missing model/calibration, low agreement, and high uncertainty. Confidence is never a synonym for model availability.

## Supported conditions

Clear chart screenshots with sufficient candle geometry, visible direction colors, and a reasonably large chart region. Android accepts platform-decodable PNG, JPEG, and WEBP through the Photo Picker/share path.

## Known failure cases and limitations

- Static screenshots cannot prove candle completion or candle progress.
- Occluding indicators, overlays, crop edges, anti-aliasing, and unusual monochrome themes can make direction ambiguous.
- A chart image does not supply reliable asset identity, price scale, payout, news, or higher-timeframe context.
- Synthetic renders validate reconstruction only and must not be mixed with real market holdouts.
- No trained artifact is included; the safe shipped behavior is `NO_TRADE`.
