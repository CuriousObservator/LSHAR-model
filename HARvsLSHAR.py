#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:51:10 2025

@author: rbarcrosspbar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import statsmodels.api as sm
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import os

warnings.filterwarnings("ignore")


# ======================================================
# Global parameters
# ======================================================

alpha = 0.1
lookback = 256
step_size = 1

# ======================================================
# Loss functions
# ======================================================

def qlike_loss(actual, forecast):
    return np.log(actual / forecast) + actual / forecast 

def rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))

# ======================================================
# RV construction
# ======================================================

def calc_daily_rv(data):
    start_time = pd.to_datetime('09:15:00').time()
    end_time = pd.to_datetime('15:30:00').time()
    in_session = data.between_time(start_time, end_time)
    rv = in_session['Ret_sq'].resample('D').sum()
    return rv[rv > 0]

# ======================================================
# Lomb–Scargle utilities
# ======================================================

def select_robust_periods(freq, power, fap, rel_tol=0.1):
    peaks, props = find_peaks(power, height=fap)
    if len(peaks) == 0:
        return []

    periods = 1.0 / freq[peaks]
    powers = power[peaks]

    df = pd.DataFrame({'period': periods, 'power': powers})
    df = df.sort_values('power', ascending=False)

    selected = []
    for p in df['period']:
        if not any(abs(p - s) / s < rel_tol for s in selected):
            selected.append(p)

    return selected


def LS_features(index, periods, scale, base_time):
    # Subtract global_start, not index[0]
    t = (index - base_time).total_seconds() / scale 
    feats = {}
    for p in periods:
        f = 1 / p
        feats[f'Sine_{p:.2f}'] = np.sin(2 * np.pi * f * t)
        feats[f'Cos_{p:.2f}'] = np.cos(2 * np.pi * f * t)
    return pd.DataFrame(feats, index=index)

def _single_max_power_sim(t, y_mean, y_std, freq, len_y):
    y_sim = y_mean + np.random.normal(0, y_std, len_y)
    model_sim = LombScargle(t, y_sim)
    power_sim = model_sim.power(freq, normalization="standard")
    return np.max(power_sim)

def mc_fap(t, y, freq, alpha, m=1000, max_workers=None):
    y_std = np.std(y)
    y_mean = np.mean(y)
    len_y = len(y)

    max_powers = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _single_max_power_sim,
                t, y_mean, y_std, freq, len_y
            )
            for _ in range(m)
        ]

        for future in as_completed(futures):
            max_powers.append(future.result())

    max_powers.sort()
    idx = int(np.floor((1 - alpha) * m))
    idx = max(0, min(idx, m - 1))
    return max_powers[idx]
# ======================================================
# Load data
# ======================================================

Tk().withdraw()
file = askopenfilename()

df = pd.read_csv(file)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime').sort_index()

df['Returns'] = np.log(df['Close']).diff()
df['Ret_sq'] = df['Returns'] ** 2
df.dropna(inplace=True)

# ======================================================
# RV features
# ======================================================

rvd = calc_daily_rv(df)
rvw = rvd.rolling(5).mean()
rvm = rvd.rolling(22).mean()

X = np.log(pd.concat([rvd, rvw, rvm], axis=1))
X.columns = ['rvd', 'rvw', 'rvm']

Y = np.log(rvd.shift(-1))

data = pd.concat([Y.rename('Y'), X], axis=1).dropna()

global_start = data.index[0]

# ======================================================
# ================= DAILY LSHAR ========================
# ======================================================

oos_HAR = {'QLIKE': {}, 'RMSE': {}}
oos_LSHAR = {'QLIKE': {}, 'RMSE': {}}
oos_Hybrid = {'QLIKE': {}, 'RMSE': {}}

start = data.index.searchsorted(pd.to_datetime('2023-01-01'))

while start + lookback < len(data):

    train = data.iloc[start:start + lookback]
    test_idx = start + lookback

    X_train = train[['rvd', 'rvw', 'rvm']]
    Y_train = train['Y']

    X_test = X.iloc[[test_idx]]
    actual_rv = rvd.iloc[test_idx]

    # HAR
    har = sm.OLS(Y_train, sm.add_constant(X_train)).fit()
    mu = har.predict(sm.add_constant(X_test, has_constant='add')).iloc[0]
    sigma2 = har.mse_resid
    har_pred = np.exp(mu + sigma2 / 2)

    oos_HAR['QLIKE'][data.index[test_idx]] = qlike_loss(actual_rv, har_pred)
    oos_HAR['RMSE'][data.index[test_idx]] = (actual_rv - har_pred) ** 2

    # Lomb–Scargle
    t = np.arange(len(X_train))   
    ls = LombScargle(t, X_train['rvd'].values)
    min_period = 5
    max_period = lookback / 2
    freq = np.linspace(1 / max_period, 1 / min_period, 2 * lookback)
    power = ls.power(freq)
    fap = ls.false_alarm_level(alpha)
    if isinstance(fap, (np.ndarray, list)):
        fap = float(np.max(fap))

    periods = select_robust_periods(freq, power, fap)

    if periods:
        LS_train = LS_features(X_train.index, periods, 86400, global_start)
        X_aug = sm.add_constant(pd.concat([X_train, LS_train], axis=1))
        lshar = sm.OLS(Y_train, X_aug).fit()

        t_oos = (data.index[test_idx] - global_start).total_seconds() / 86400
        LS_oos = LS_features(pd.Index([data.index[test_idx]]),periods,86400, global_start).reset_index(drop=True)

        X_oos = pd.concat([X_test.reset_index(drop=True), LS_oos], axis=1)
        X_oos = sm.add_constant(X_oos, has_constant='add')
        X_oos.columns = lshar.params.index

        mu_ls = lshar.predict(X_oos).iloc[0]
        sigma2_ls = lshar.mse_resid
        lshar_pred = np.exp(mu_ls + sigma2_ls / 2)

        oos_LSHAR['QLIKE'][data.index[test_idx]] = qlike_loss(actual_rv, lshar_pred)
        oos_LSHAR['RMSE'][data.index[test_idx]] = (actual_rv - lshar_pred) ** 2

        hybrid_pred = lshar_pred
    else:
        hybrid_pred = har_pred

    oos_Hybrid['QLIKE'][data.index[test_idx]] = qlike_loss(actual_rv, hybrid_pred)
    oos_Hybrid['RMSE'][data.index[test_idx]] = (actual_rv - hybrid_pred) ** 2

    start += step_size
    
print("\n" + "="*60)
display = os.path.basename(file).split('_')[0]
print(f"ANALYSIS FOR {display} STOCK.")
print("\n" + "="*60)
print("DAILY FORECASTING RESULTS")
print("="*60)

har_qlike = np.mean(list(oos_HAR['QLIKE'].values()))
har_rmse  = np.sqrt(np.mean(list(oos_HAR['RMSE'].values())))

hyb_qlike = np.mean(list(oos_Hybrid['QLIKE'].values()))
hyb_rmse  = np.sqrt(np.mean(list(oos_Hybrid['RMSE'].values())))

print(f"HAR     | QLIKE: {har_qlike:.6f} | RMSE: {har_rmse:.6f}")
print(f"HYBRID  | QLIKE: {hyb_qlike:.6f} | RMSE: {hyb_rmse:.6f}")

if oos_LSHAR['QLIKE']:
    lshar_qlike = np.mean(list(oos_LSHAR['QLIKE'].values()))
    lshar_rmse  = np.sqrt(np.mean(list(oos_LSHAR['RMSE'].values())))
    # print(f"LSHAR   | QLIKE: {lshar_qlike:.6f} | RMSE: {lshar_rmse:.6f}")
    # print(f"LSHAR ACTIVE DAYS: {len(oos_LSHAR['QLIKE'])}")
else:
    print("LSHAR never activated in daily setting.")
    
if oos_LSHAR['QLIKE']:
    har_filtered_qlike = np.mean([
        oos_HAR['QLIKE'][d] for d in oos_LSHAR['QLIKE'].keys()
    ])
    har_filtered_rmse = np.sqrt(np.mean([
    oos_HAR['RMSE'][d] for d in oos_LSHAR['RMSE'].keys()
    ]))

    print("\nAPLES-TO-APPLES (LS ACTIVE DAYS ONLY)")
    print(f"HAR (filtered) QLIKE:  {har_filtered_qlike:.6f}")
    print(f"LSHAR QLIKE:          {lshar_qlike:.6f}")
    print(f"HAR (filtered) RMSE:  {har_filtered_rmse:.6f}")
    print(f"LSHAR RMSE:          {lshar_rmse:.6f}")

# ======================================================
# ================= WEEKLY LSHAR =======================
# ======================================================

oos_weekly_HAR = {'QLIKE': {}, 'RMSE': {}}
oos_weekly_LSHAR = {'QLIKE': {}, 'RMSE': {}}
oos_weekly_Hybrid = {'QLIKE': {}, 'RMSE': {}}
lookback_weeks = 52*5
weekly_step_size = 5
start = data.index.searchsorted(pd.to_datetime('2023-01-01'))

while start + lookback_weeks < len(data):

    train = data.iloc[start:start + lookback_weeks]
    test_idx = start + lookback_weeks

    X_train = train[['rvd', 'rvw', 'rvm']]
    Y_train = train['Y']

    X_test = X.iloc[[test_idx]]
    actual_rv = rvd.iloc[test_idx]

    har = sm.OLS(Y_train, sm.add_constant(X_train)).fit()
    mu = har.predict(sm.add_constant(X_test, has_constant='add')).iloc[0]
    sigma2 = har.mse_resid
    har_pred = np.exp(mu + sigma2 / 2)

    t = np.arange(len(X_train)) / 5.0
    ls = LombScargle(t, X_train['rvw'].values)
    min_period = 2
    max_period = (lookback_weeks / 5) / 2  # in weeks
    freq = np.linspace(1 / max_period, 1 / min_period, 2 * lookback_weeks)
    power = ls.power(freq)
    fap = ls.false_alarm_level(alpha)
    if isinstance(fap, (np.ndarray, list)):
        fap = float(np.max(fap))

    periods = select_robust_periods(freq, power, fap)

    if periods:
        LS_train = LS_features(X_train.index, periods, 604800, global_start)
        X_aug = sm.add_constant(pd.concat([X_train, LS_train], axis=1))
        model = sm.OLS(Y_train, X_aug).fit()
    
        t_oos = (data.index[test_idx] - global_start).total_seconds() / 604800
        LS_oos = LS_features(pd.Index([data.index[test_idx]]), periods, 604800, global_start)
    
        X_oos = pd.concat(
            [X_test.reset_index(drop=True), LS_oos.reset_index(drop=True)],
            axis=1
        )
        X_oos = sm.add_constant(X_oos, has_constant='add')
        X_oos.columns = model.params.index
    
        mu_h = model.predict(X_oos).iloc[0]
        sigma2_h = model.mse_resid
        lshar_pred = np.exp(mu_h + sigma2_h / 2)
    
        # LSHAR active
        oos_weekly_LSHAR['QLIKE'][data.index[test_idx]] = qlike_loss(actual_rv, lshar_pred)
        oos_weekly_LSHAR['RMSE'][data.index[test_idx]] = (actual_rv - lshar_pred) ** 2
    
        hybrid_pred = lshar_pred
    else:
        hybrid_pred = har_pred
    
    oos_weekly_HAR['QLIKE'][data.index[test_idx]] = qlike_loss(actual_rv, har_pred)
    oos_weekly_HAR['RMSE'][data.index[test_idx]] = (actual_rv - har_pred) ** 2
# Hybrid always recorded
    oos_weekly_Hybrid['QLIKE'][data.index[test_idx]] = qlike_loss(actual_rv, hybrid_pred)
    oos_weekly_Hybrid['RMSE'][data.index[test_idx]] = (actual_rv - hybrid_pred) ** 2
    start += weekly_step_size
    
print("\n" + "="*60)
print("WEEKLY FORECASTING RESULTS")
print("="*60)

har_qlike = np.mean(list(oos_weekly_HAR['QLIKE'].values()))
har_rmse  = np.sqrt(np.mean(list(oos_weekly_HAR['RMSE'].values())))

hyb_qlike = np.mean(list(oos_weekly_Hybrid['QLIKE'].values()))
hyb_rmse  = np.sqrt(np.mean(list(oos_weekly_Hybrid['RMSE'].values())))

print(f"HAR     | QLIKE: {har_qlike:.6f} | RMSE: {har_rmse:.6f}")
print(f"HYBRID  | QLIKE: {hyb_qlike:.6f} | RMSE: {hyb_rmse:.6f}")

if oos_weekly_LSHAR['QLIKE']:
    lshar_qlike = np.mean(list(oos_weekly_LSHAR['QLIKE'].values()))
    lshar_rmse  = np.sqrt(np.mean(list(oos_weekly_LSHAR['RMSE'].values())))
    print(f"LSHAR   | QLIKE: {lshar_qlike:.6f} | RMSE: {lshar_rmse:.6f}")
    print(f"LSHAR ACTIVE WEEKS: {len(oos_weekly_LSHAR['QLIKE'])}")
else:
    print("LSHAR never activated in weekly setting.")
# ======================================================
# ===== RESIDUAL (ORTHOGONAL) SPECTRAL AUGMENTATION =====
# ======================================================

results = []
LS_periods = 0
start = data.index.searchsorted(pd.to_datetime('2023-01-01'))

while start + lookback < len(data):

    train = data.iloc[start:start + lookback]
    test_idx = start + lookback

    X_train = train[['rvd', 'rvw', 'rvm']]
    Y_train = train['Y']

    X_test = X.iloc[[test_idx]]
    actual_rv = rvd.iloc[test_idx]

    har = sm.OLS(Y_train, sm.add_constant(X_train)).fit()
    mu = har.predict(sm.add_constant(X_test, has_constant='add')).iloc[0]
    sigma2 = har.mse_resid
    har_pred = np.exp(mu + sigma2 / 2)

    resid = Y_train - har.fittedvalues

    t = np.arange(len(resid))
    ls = LombScargle(t, resid.values)
    min_period = 5
    max_period = lookback / 2
    freq = np.linspace(1 / max_period, 1 / min_period, 2 * lookback)
    power = ls.power(freq)
    fap = ls.false_alarm_level(alpha)
    if isinstance(fap, (np.ndarray, list)):
        fap = float(np.max(fap))

    periods = select_robust_periods(freq, power, fap)

    hybrid_pred = har_pred

    if periods:
        LS_periods += 1
        LS_train = LS_features(X_train.index, periods, 86400, global_start)
        model = sm.OLS(resid, sm.add_constant(LS_train)).fit()

        t_oos = (data.index[test_idx] - global_start).total_seconds() / 86400
        LS_oos = LS_features(pd.Index([data.index[test_idx]]), periods, 86400, global_start)
        resid_pred = model.predict(sm.add_constant(LS_oos, has_constant='add')).iloc[0]
        hybrid_pred = np.exp(mu + resid_pred + sigma2 / 2)

    results.append({
        'Date': data.index[test_idx],
        'Actual': actual_rv,
        'HAR': har_pred,
        'Hybrid': hybrid_pred
    })

    start += step_size

# ======================================================
# Final diagnostics
# ======================================================
print("\n" + "="*60)
print("RESIDUALS MODELLING RESULTS")
print("="*60)
res = pd.DataFrame(results).set_index('Date')
if(LS_periods<0.5*len(results)):
    print("No significant LS_periods were found, in the residuals.")
else:
    print("\nResidual-Augmented HAR")
    print("HAR QLIKE:", qlike_loss(res['Actual'], res['HAR']).mean())
    print("Hybrid QLIKE:", qlike_loss(res['Actual'], res['Hybrid']).mean())
    print("HAR RMSE:", rmse(res['Actual'], res['HAR']))
    print("Hybrid RMSE:", rmse(res['Actual'], res['Hybrid']))
    print("LS activations:", LS_periods)

print("="*60)