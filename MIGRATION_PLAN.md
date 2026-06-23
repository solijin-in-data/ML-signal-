# MIGRATION PLAN

Phase 1 creates a safe scaffold only. It does not rewrite the core logic.

Run:

```bash
python scaffold_refactor_project.py
python scaffold_refactor_project.py --apply
```

After that, confirm the old commands still work.

Then archive patch scripts:

```bash
python scaffold_refactor_project.py --apply --archive-patches
```

Suggested validation command:

```bash
python feature_experiment_runner.py --ticker CTD --profile SWING --mode fixed --feature-set ablation_trend_recovery_minus_market_regime_v1 --lookahead 30 --tp 0.08 --sl -0.05 --threshold 0.55 --min-edge-vs-breakeven 0.02 --no-diagnostics-fallback
```
