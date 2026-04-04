# Lomb-Scargle HAR (LSHAR): Spectral Augmentation

This repository implements the **Three-Gate LSHAR Framework** as detailed in the study *"When Does HAR Fail? Tactical Spectral Augmentation via Lomb-Scargle Periodograms"*. 

The project introduces a selective regime-detection strategy that identifies when the standard Heterogeneous Autoregressive (HAR) model fails to capture transient periodicities in realized volatility.

## Project Overview

Traditional volatility models like the HAR rely on discrete "box" kernels (daily, weekly, monthly) which assume a static memory structure. This project proposes that volatility periodically enters **"Active Regimes"** where sinusoidal rhythms emerge. 

We use the **Lomb-Scargle Periodogram (LSP)**—an astrophysical signal processing tool—to identify these rhythms in irregularly sampled financial time series (accounting for weekends and holidays) without the need for interpolation or "gap-filling" which can bias spectral estimates.

## The Three-Gate Validation Logic

To resolve the "Hybrid Paradox" (where switching models introduces noise and estimation variance), this implementation employs a rigorous selective activation process:

1.  **Gate 1: Spectral Discovery:** Uses the LSP to identify potential frequencies in Daily Realized Volatility ($RV$).
2.  **Gate 2: Significance Testing:** Employs a **Moving Block Bootstrap** to calculate False Alarm Probability (FAP), ensuring detected signals are not stochastic artifacts.
3.  **Gate 3: Persistence Filter:** Only activates the LSHAR model if a frequency remains stable across $k=3$ consecutive rolling windows, ensuring structural reliability.

## Core Findings ($N=99$ NSE Assets)

* **Quality over Quantity:** Sparse, high-confidence spectral interventions (active $<5\%$ of days) deliver a **0.358% QLIKE improvement**, over twice the gain of frequent augmentations ($>15\%$ activity: 0.148%).
* **Surgical Precision:** Low-activity interventions exhibit a **76% success rate**, outperforming high-activity models by a factor of **2.43x**.
* **Model Interpretation:** Empirical results confirm that model activity is inversely proportional to predictive value. The LSHAR's value lies in surgical regime detection rather than persistent structural modeling.
* **Spectral Whitening:** As shown in the study's residual analysis, the LSHAR successfully "whitens" the spectral peaks that the standard HAR model leaves behind, neutralizing unexplained periodic memory.

## Requirements

* **Python 3.x**
* **Astropy:** For the Lomb-Scargle implementation.
* **Statsmodels:** For OLS/WLS fitting and diagnostic testing.
* **Pandas, NumPy, Matplotlib.**
