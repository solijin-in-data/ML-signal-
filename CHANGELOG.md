# Changelog

All notable changes to this project will be documented in this file.

This project follows a research-oriented development process. Changes are grouped by version and by development area.

## [Unreleased]

### Planned

* Add feature correlation analysis module.
* Add multi-scale `n` testing for EMA, momentum, and efficiency ratio.
* Add feature ablation testing.
* Add market regime filter based on VNINDEX trend and volatility.
* Add T+ settlement-aware target labeling.
* Add OHLC-based take-profit and stop-loss detection.
* Add transaction cost and slippage assumptions.
* Add trade-level backtest report.
* Add signal stability report across walk-forward folds.
* Add permutation importance for feature explainability.
* Refactor the project into a modular `src/` architecture.

## [0.1.0] - Initial Research Version

### Added

* Added raw market data standardization pipeline.
* Added automatic file type detection for stock files and VNINDEX benchmark file.
* Added special VNINDEX schema handling:

  * `VN_Open`
  * `VN_High`
  * `VN_Low`
  * `VN_Close`
  * `VN_Volume`
  * `VN_Change_Pct`
* Added standardized stock schema for OHLCV and foreign trading flow.
* Added automatic date parsing for both Vietnamese and US date formats.
* Added numeric parser for values containing commas, percentages, and K/M/B suffixes.
* Added audit report for data standardization.
* Added machine learning signal radar pipeline.
* Added feature engineering for:

  * Log return
  * RSI
  * Momentum
  * EMA structure
  * MACD
  * Bollinger distance
  * Volatility
  * VNINDEX return
  * VNINDEX volatility
  * Relative strength
  * Foreign net flow
* Added binary target creation using lookahead, take-profit, and stop-loss rules.
* Added walk-forward validation using `TimeSeriesSplit`.
* Added XGBoost classifier for signal prediction.
* Added probability threshold optimization.
* Added final signal classification:

  * `STRONG WATCH`
  * `VALID ENTRY`
  * `WEAK SIGNAL`
  * `NO TRADE`
* Added evaluation chart export.
* Added signal summary CSV export.
* Added capped Half-Kelly position sizing indicator.

### Changed

* Replaced hard-coded raw data paths with project-based processed data paths.
* Replaced console `print()` messages with Python `logging`.
* Converted logs and runtime messages to English.
* Removed emojis from terminal output.
* Changed output structure to use the `outputs/` folder.
* Changed chart output location to `outputs/charts/`.
* Standardized time-series order to oldest date first.
* Improved latest signal generation by predicting from the latest available feature row instead of the last labeled row.

### Fixed

* Fixed the issue where the latest signal was generated from a delayed row due to target creation removing the most recent unlabeled observations.
* Fixed VNINDEX being incorrectly treated as a normal stock file.
* Fixed date parsing warnings caused by mixed Vietnamese and US date formats.
* Fixed potential time-order issues by enforcing ascending date order before feature engineering and target creation.

### Known Issues

* Current labeling still uses close-price movement instead of OHLC high-low path.
* Current backtest is closer to signal validation than full trade execution simulation.
* T+ settlement rules are not fully modeled.
* Transaction costs and slippage are not fully included.
* Feature redundancy and correlation pruning are not yet implemented.
* Model probability is not yet calibrated.
* External macro, policy, and geopolitical shocks are not directly modeled.
