# Reproducibility Notes

## Production signal

Run from project root:

```bash
python scripts/production/run_signal.py ^
  --ticker=CTD ^
  --profile=SWING ^
  --config=configs/experiments/ctd_cost_resilient_swing_v1.yaml
```

Generated files:

```text
reports/signals/CTD_SWING_latest_signal.md
reports/signals/CTD_SWING_latest_signal.json
```

## Current production-candidate setup

```text
ticker: CTD
profile: SWING
candidate: production_candidate_cost_resilient_v1
feature_set: candidate_cost_resilient_recovery_no_noise_v1
lookahead: 40
tp: 0.10
sl: -0.05
threshold: 0.60
stress_round_trip_cost: 0.005
```

## Data assumptions

The project expects standardized market data under:

```text
data/processed_data/
```

Example expected files:

```text
CTD_standardized.csv
VNINDEX_standardized.csv
```

Valuation data, when used, is expected under:

```text
data/processed_data/valuation/
```

## Git hygiene

Signal reports can change when the production runner is executed because timestamps are regenerated. Restore them unless the report update is intentional:

```bash
git restore reports/signals/CTD_SWING_latest_signal.json
git restore reports/signals/CTD_SWING_latest_signal.md
```
