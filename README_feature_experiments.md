# Feature Experiment Runner

This patch adds `feature_experiment_runner.py`.

The purpose is to test whether new feature groups should be promoted into the default model.

## Main idea

The runner compares:

- `baseline`
- `candidate_momentum_v1`
- `candidate_trend_quality_v1`
- `candidate_volume_v1`
- `candidate_recovery_v1`
- `candidate_light_combo_v1`

It supports two experiment modes:

- `fixed`: compare feature sets using the same existing setup.
- `reoptimized`: allow each feature set to search across the profile grid again.
- `both`: run both modes.

## Recommended first command

```bash
python feature_experiment_runner.py --ticker CTD --profile SWING --mode both
```

Because CTD currently has no valid `best_params.csv`, the runner can automatically fall back to:

```text
outputs/setup_diagnostics/CTD_SWING_setup_diagnostics.csv
```

and use the highest-score setup as the fixed setup.

## Explicit fixed setup example

```bash
python feature_experiment_runner.py --ticker CTD --profile SWING --mode fixed --lookahead 20 --tp 0.10 --sl -0.07 --threshold 0.50
```

## Output

The runner writes:

```text
outputs/feature_experiments/
├── feature_set_comparison.csv
├── feature_health_report.csv
└── feature_set_decision_report.csv
```

## Important columns

In `feature_set_comparison.csv`, focus on:

- `Edge_vs_Breakeven`
- `Avg_Return`
- `Expectancy_Per_Day`
- `Timeout_Rate`
- `Trades`
- `Score_Delta_vs_Baseline`
- `Edge_vs_Breakeven_Delta_vs_Baseline`
- `Avg_Return_Delta_vs_Baseline`

A candidate feature set should not be accepted just because precision improves. It should improve edge, return, and expectancy without making holding period or timeout risk much worse.
