# Lomb-Scargle HAR (LSHAR) for Volatility Prediction

This repository implements the **Lomb-Scargle Heterogeneous Autoregressive (LSHAR)** model as detailed in the study "Improving the standard HAR model for volatility prediction using Lomb-Scargle Periodicities". The project evaluates how spectral features from non-uniformly sampled financial data can enhance traditional volatility forecasting.

## Project Overview

The project extends the standard HAR model—which uses daily, weekly, and monthly lags to represent different market participants—by incorporating periodic signals identified via the **Lomb-Scargle Periodogram (LSP)**. The LSP is specifically used to handle the irregular sampling inherent in market data due to weekends and holidays.

## Key Features

* **Spectral Augmentation:** Identifies hidden periodicities in Realized Volatility (RV) data that are not captured by traditional linear lags.


* **Statistical Significance:** Employs False Alarm Probability (FAP) thresholds, calculated via Monte-Carlo simulations or analytical methods, to filter significant frequencies.


* **Multi-Frequency Evaluation:** Provides a comparative analysis of model performance on both daily and weekly data resolutions.


* **Validation of Efficiency:** Includes a residual analysis to test if the standard HAR model leaves any systematic periodic information uncaptured.



## Core Findings

* **Daily Advantage:** Standalone LSHAR modeling on daily data consistently outperforms the HAR baseline in predictive precision (QLIKE) and error reduction (RMSE).


* **Aggregation Loss:** The predictive edge of spectral features diminishes at a weekly scale as data aggregation smooths out high-frequency rhythms.


* **Spectral Efficiency:** Residual tests confirm that the standard HAR model is an efficient filter, as its residuals typically show no significant periodic signals.



## Requirements

* **Python 3.x**
* **Astropy:** For Lomb-Scargle implementation.


* **Statsmodels:** For OLS fitting and HAR baseline construction.


* **Pandas & NumPy:** For data manipulation and RV calculations.


* **Matplotlib:** For periodogram and volatility trend visualization.



## Usage

1. **Data Preparation:** Clean raw OHLC data using the provided `clean_csv.py` to ensure proper Indian market timestamps.


2. **Feature Engineering:** Generate daily, weekly, and monthly RV components.


3. **Model Execution:** Run the walk-forward analysis to evaluate the HAR, LSHAR, and Hybrid models across rolling windows.



## Author

**MVR Abhishek** 

January 2026
