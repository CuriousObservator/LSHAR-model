# LSHAR Model — Lomb–Scargle Augmented HAR for NSE Realized Volatility

**Paper:** *Scale-Specific Periodic Structure in NSE Realized Volatility: Evidence from Lomb–Scargle Analysis Across 99 Equity Assets*
**Author:** Abhishek M V R
**Status:** Preprint, April 2026

---

## Overview

This repository contains the full implementation of the LSHAR (Lomb–Scargle Heterogeneous Autoregressive) framework, which detects and exploits persistent periodic structure in NSE realized volatility using a spectral estimator designed for irregularly sampled time series.

The central finding is that periodic volatility structure in NSE equities is **asset-specific in its timescale**: a daily-resolution Lomb–Scargle detector and a weekly-aggregated detector identify largely non-overlapping sets of stocks (activation correlation ≈ 0.00). Rather than positioning the detected cycles as a universal HAR replacement, the framework is best interpreted as a **volatility regime filter** — a binary signal identifying when a stock's realized volatility is governed by a recurring periodic component, with direct implications for options pricing and volatility-sensitive strategies.

---

## Repository Structure

```
.
├── HARvsLSHAR.py                        # Main implementation (see below)
├── csv_cleaner.py                       # Data preprocessing
├── LSHAR_Publication_Results2.csv       # Full per-asset results, k=2
├── LSHAR_Publication_Results3.csv       # Full per-asset results, k=3 (baseline)
├── LSHAR_Publication_Results4.csv       # Full per-asset results, k=4
└── README.md
```

---

## Data

**Source:** NSE 5-minute OHLC data via [Kaggle — NSE Nifty-100 5-minute data](https://www.kaggle.com/datasets/debashis74017/stock-market-data-nifty-100-stocks-5-min-data)

- 99 liquid NSE equities including NIFTY 50 and NIFTY Bank constituents
- ~640 trading days, early 2021 through late 2024
- Market hours: 09:15–15:30 IST; broken/thin-trading sessions removed
- INDIA VIX is included in the spectral analysis but excluded from the 99-asset forecast evaluation (not a tradeable equity)

Preprocess raw CSVs with:
```bash
python csv_cleaner.py
```

---

## Implementation — `HARvsLSHAR.py`

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `ALPHA` | `0.05` | FAP significance level |
| `M_SIMULATIONS` | `1000` | Bootstrap replications for FAP threshold |
| `LOOKBACK_DAILY` | `252` | Rolling window for HAR / daily LS (trading days) |
| `LOOKBACK_WEEKLY_LS` | `520` | Lookback for weekly aggregated LS (~104 weekly observations) |
| `PERSISTENCE_WINDOW` | `4` | Consecutive detections required before activation — set to 2, 3, or 4 |
| `PERIOD_TOLERANCE` | `0.15` | Max coefficient of variation for period stability (15%) |

To reproduce baseline paper results set `PERSISTENCE_WINDOW = 3`. The three results CSVs correspond to runs with `PERSISTENCE_WINDOW` set to 2, 3, and 4 respectively.

### Three spectral tracks

The code runs three parallel detection tracks for each stock:

**1. LSHAR (daily)** — Lomb–Scargle periodogram on log daily RV using a 252-day rolling window. Augments the HAR regression with a sin/cos Fourier pair when a stable period passes all three gates.

**2. ResidLS (daily, residual domain)** — Periodogram applied to HAR residuals rather than raw log RV. Included for comparison; this track is structurally fragile (L/G = 2.85× at k=3) and is not recommended for practical use.

**3. WeeklyLS** — Lomb–Scargle periodogram on non-overlapping 5-day summed RV blocks using a 520-day lookback. Detected periods are converted back to trading days (×5) before constructing spectral features. This is the most stable track across all values of k.

### Three-gate detection pipeline

Each track gates spectral augmentation sequentially:

1. **Gate 1 — Spectral discovery:** LSP computed over a log-spaced frequency grid (100 points, f_min = 1/T to pseudo-Nyquist).
2. **Gate 2 — Bootstrap FAP:** Moving block bootstrap (block length = ⌊√n⌋, 1000 replications). Threshold recomputed when rolling-window volatility std shifts >20% or every 30 iterations.
3. **Gate 3 — Persistence filter (`PeriodTracker`):** Period must remain stable (CV < 15%) across `PERSISTENCE_WINDOW` consecutive windows. A single missing detection resets the history. The activation of this gate is the regime-entry signal.

### HAR augmentation model

When a period `p` clears all three gates:

```
log(RV_{t+1}) = β₀ + βd·log(RVd_t) + βw·log(RVw_t) + βm·log(RVm_t)
              + γs·sin(2π/p̂_t · (t+1)) + γc·cos(2π/p̂_t · (t+1)) + ε
```

When no period survives, the model reverts to standard log-HAR. Forecasts apply a Jensen's correction on back-transformation: `RV_hat = exp(log_RV_hat + σ²_resid / 2)`.

Performance is evaluated with **QLIKE loss**, computed globally and conditionally on active days. The conditional improvement metric is:

```
Δ = (QLIKE_HAR - QLIKE_model) / |QLIKE_HAR| × 100%
```

### Running the analysis

```bash
python HARvsLSHAR.py
```

A folder picker dialog will open. Select the directory containing your cleaned CSVs. The script uses `ProcessPoolExecutor` for parallelism across stocks and outputs `LSHAR_Publication_Results{k}.csv` to the same folder.

---

## Results Summary

All figures below use the baseline `k = 3`. Full per-asset metrics for `k ∈ {2, 3, 4}` are in the results CSVs.

### Aggregate track performance (k=3, 99 assets ex-VIX)

| Track | Active stocks | Win rate | Mean Δ (active days) | Loss/Gain ratio |
|---|---|---|---|---|
| Daily LSHAR | 89/99 | 21% | −0.25% | 0.81× |
| ResidLS | 59/99 | 36% | −0.63% | 2.85× |
| **WeeklyLS** | **73/99** | **33%** | **−0.06%** | **0.76×** |

Loss/Gain ratio below 1× means magnitude-weighted gains outweigh losses among active stocks. WeeklyLS maintains sub-unity L/G at all three values of k (0.40× to 0.76×).

### Scale-specificity (k=3)

The daily and weekly detectors are near-orthogonal (activation correlation ≈ 0.00):

| Pattern | Stocks | Mean Δ |
|---|---|---|
| Daily only fires and wins | 14 | +0.60% |
| Weekly only fires and wins | 19 | +0.39% |
| Both fire, both win | 5 | +0.34% |
| Both fire, neither wins | 34 | −0.35% |
| Neither fires (CANBK) | 1 | — |

### Best-performing stocks across k

| Stock | Track | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| INFY | WeeklyLS | +2.34% | +2.07% | −0.17% |
| TRENT | LSHAR | +1.55% | +1.74% | +1.70% |
| HDFCLIFE | WeeklyLS | +0.99% | +1.01% | +1.07% |
| ADANIGREEN | WeeklyLS | +0.19% | +0.86% | +0.61% |
| AXISBANK | LSHAR | +0.73% | +0.91% | +0.90% |
| PIDILITIND | LSHAR | +0.94% | +0.87% | +0.90% |

### Robustness across persistence window k

| Track | k | Active | Win% | Mean Δ | L/G |
|---|---|---|---|---|---|
| Daily LSHAR | 2 | 90 | 20% | −0.24% | 0.91× |
| | 3 | 89 | 21% | −0.25% | 0.81× |
| | 4 | 86 | 21% | −0.20% | 0.59× |
| ResidLS | 2 | 69 | 29% | −0.40% | 1.46× |
| | 3 | 59 | 36% | −0.63% | 2.85× |
| | 4 | 54 | 37% | −0.58% | 3.48× |
| **WeeklyLS** | 2 | 80 | 32% | +0.01% | **0.45×** |
| | 3 | 73 | 33% | −0.06% | **0.76×** |
| | 4 | 63 | 33% | +0.04% | **0.40×** |

Total assets with at least one activation: 99 at k=2, 98 at k=3, 95 at k=4.

---

## Interpreting the Regime Signal

The persistence gate activation is best understood as a **volatility regime classifier** rather than a point-forecast improvement. When the gate fires:

- The stock's realized volatility is temporarily governed by a recurring periodic component.
- The sub-unity L/G ratio means magnitude-weighted outcomes favour the augmented model on active days.
- The signal is most actionable for mid-cap and sector-specific stocks with moderate analyst coverage — stocks where options markets are least likely to have already priced the cycle in.

**Practical guidance on k:**
- `k=2` — broader coverage, more noise-driven activations; suitable if recall matters more than precision
- `k=3` — recommended default; reasonable balance for both daily and weekly tracks
- `k=4` — cleaner daily LSHAR activations (L/G improves to 0.59×) but can eliminate genuine quarterly WeeklyLS cycles (INFY loses activation entirely); use with care for weekly track

**High-activation warning:** Stocks where WeeklyLS activates on >15% of test days tend to show near-zero improvement on average. A signal that is "always on" has no discriminatory regime value. Consider capping the maximum activation duration in production use.

**ResidLS:** The residual-domain track is included for completeness and reproducibility but should not be used operationally. Its L/G degrades monotonically with stricter k (1.46× → 2.85× → 3.48×), indicating structural fragility.

---

## Dependencies

```
numpy
pandas
statsmodels
astropy
scipy
tqdm
tkinter   # standard library, for folder picker
```

Install with:
```bash
pip install numpy pandas statsmodels astropy scipy tqdm
```

---

## Acknowledgements

The author thanks Jose Carlos Gonzales Tanaka for substantive discussions that shaped the direction of this work.
