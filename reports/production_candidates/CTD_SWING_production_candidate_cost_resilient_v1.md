# CTD SWING Production Candidate Report

**Candidate:** `production_candidate_cost_resilient_v1`

**Decision:** `PROMOTE_TO_PRODUCTION_CANDIDATE`

**Reason:** Frozen grid evidence satisfies the acceptance rules under the selected cost assumption.

## Recommended setup

| Metric | Value |
| --- | --- |
| Feature Set | candidate_cost_resilient_recovery_no_noise_v1 |
| Frozen Grid Source Set | ablation_cost_resilient_recovery_minus_noise_filter_v1 |
| LOOKAHEAD | 40 |
| TP | 10.00% |
| SL | -5.00% |
| THRESH | 60.00% |
| Round-trip Cost | 0.50% |
| Periods | 4 |
| Pass Periods | 3 |
| Pass Rate | 75.00% |
| Avg Edge | 11.23% |
| Min Edge | 7.33% |
| Avg Return | 3.48% |
| Min Return | 2.65% |
| Total Trades | 887 |
| Frozen Strict Candidate | True |

## Acceptance checks

| Rule | Observed | Status |
| --- | --- | --- |
| Pass_Periods >= 3 | 3 | PASS |
| Min_Edge >= 2.00% | 7.33% | PASS |
| Avg_Return >= 2.00% | 3.48% | PASS |
| Total_Trades >= 300 | 887 | PASS |

## Research interpretation

The strongest frozen-grid setup excludes the noise-filter feature family. This suggests the production candidate should retain liquidity and recovery-quality features, while keeping noise-filter features only for research and ablation tests.

## Reproducibility command

```cmd
python frozen_parameter_grid_runner.py ^
  --ticker=CTD ^
  --profile=SWING ^
  --feature-sets=candidate_cost_resilient_recovery_no_noise_v1 ^
  --lookaheads=40 ^
  --tps=0.10 ^
  --sls=-0.05 ^
  --thresholds=0.60 ^
  --round-trip-cost=0.005 ^
  --quiet
```

## Source

- Source CSV: `C:\Users\tuana\OneDrive\Documents\GitHub\ML-signal-\outputs\walkforward_frozen_grid\CTD_SWING_frozen_grid_stability_score.csv`
- Source modified: `2026-06-23T14:23:01`
- Report generated: `2026-06-23T14:35:14`
