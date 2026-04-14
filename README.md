# Lomb–Scargle HAR (LSHAR): Spectral Augmentation of Realized Volatility

This repository implements the **Three-Gate LSHAR Framework** as detailed in the study *"Scale-Specific Periodic Structure in NSE Realized Volatility: Evidence from Lomb–Scargle Analysis Across 99 Equity Assets"*.

The project introduces a selective spectral augmentation of the standard HAR model that detects and exploits periodic structure in realized volatility, operating at both daily and weekly aggregation scales.

## Project Overview

Traditional volatility models like HAR rely on fixed discrete horizons (daily, weekly, monthly), imposing a static memory structure. Periodicities outside these bands — quarterly earnings cycles, options expiry effects, sector-specific rebalancing windows — are invisible to the model by construction.

We use the **Lomb–Scargle Periodogram (LSP)** — an astrophysical signal processing tool — to identify these rhythms in irregularly sampled financial time series, accounting for weekends and holidays without interpolation or gap-filling, which can bias spectral estimates.

A key empirical finding is **scale-specificity**: the daily and weekly spectral detectors identify largely non-overlapping sets of assets (activation correlation ρ = −0.10), suggesting that periodic volatility structure reflects firm-level information dynamics rather than uniform market-wide calendar effects.

## The Three-Gate Validation Logic

To avoid the noise introduced by indiscriminate model switching, spectral augmentation is gated through three sequential tests:

1. **Gate 1: Spectral Discovery** — LSP identifies candidate frequencies in realized volatility over a data-driven log-spaced frequency grid.
2. **Gate 2: Significance Testing** — A **Moving Block Bootstrap** calibrates a False Alarm Probability (FAP) threshold, ensuring detected signals are not stochastic artefacts of GARCH-type clustering.
3. **Gate 3: Persistence Filter** — A frequency is admitted only if it remains stable (CV < 15%) across k = 3 consecutive rolling windows, ensuring structural reliability before augmenting the model.

When a period passes all three gates, the HAR regression is augmented with a sine/cosine Fourier pair at the detected period. When no period survives, the model reverts to standard HAR.

## Weekly Extension

A parallel spectral track operates on weekly-aggregated realized volatility (non-overlapping 5-day sums) using an extended lookback of ~104 weekly observations. This filters intraday microstructure noise that contaminates daily periodograms, recovering lower-frequency cycles — quarterly rebalancing, semi-annual earnings seasonality — that are submerged in daily-resolution data. The weekly track is the stronger performer: its loss–gain ratio falls below unity, meaning gains outweigh losses in expectation when the model activates.

## Core Findings (N = 99 NSE Assets)

- **90 of 99 assets** exhibit at least one statistically significant periodic component at daily or weekly scale under block-bootstrap FAP testing.
- Daily and weekly detectors are **near-orthogonal** (ρ = −0.10), identifying largely non-overlapping asset subsets — stable across persistence windows k ∈ {2, 3, 4}.
- **Weekly spectral features outperform daily** on win rate, mean QLIKE improvement, and loss–gain ratio. k = 3 is the practical optimum; performance degrades at k = 4.
- Assets with the most robust improvement include **HDFCLIFE**, **TRENT**, **KOTAKBANK**, and **INFY**, each driven by cycles at timescales well outside HAR's three characteristic horizons.
- Augmentation is most reliable in the **moderate activation regime** (5–15% of days). High-activation cases degrade due to collinearity with HAR's monthly component.

## Requirements

- **Python 3.x**
- **Astropy** — Lomb–Scargle implementation
- **Statsmodels** — OLS fitting
- **Pandas, NumPy, Matplotlib**

## Repository Structure

```
├── HARvsLSHAR.py                                        # Main implementation
├── csv_cleaner.py                                       # Data preprocessing
├── LSHAR Publication Results.csv                        # Per-asset metrics, k=3 baseline
├── LSHAR Publication Results persistence {2,3,4}.csv    # Robustness results across k
└── README.md
```

Data: NSE 5-minute OHLC for 99 liquid equities available on Kaggle: [NSE Nifty-100 5-minute data](https://www.kaggle.com/).
