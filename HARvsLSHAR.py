#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 01:13:55 2026

@author: rbarcrosspbar
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import os
import tkinter as tk
from tkinter import filedialog
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple, Optional
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")


ALPHA = 0.05                
M_SIMULATIONS = 1000        
LOOKBACK_DAILY = 252        
LOOKBACK_WEEKLY_LS = 520    
PERSISTENCE_WINDOW = 3 # can be set between 2, 3 and 4      
PERIOD_TOLERANCE = 0.15     



def safe_log(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, floor))

def qlike_loss(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    forecast = np.maximum(forecast, 1e-10)
    return np.log(forecast) + (actual / forecast)


def get_data_driven_freqs(t: np.ndarray, n_freqs: int = 100) -> np.ndarray:
    T = t.max() - t.min()
    N = len(t)
    f_min = 1.0 / T
    avg_sampling_rate = N / T
    f_max = 0.5 * avg_sampling_rate
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
    return freqs


def get_bootstrap_threshold(
    t: np.ndarray,
    residuals: np.ndarray,
    freqs: np.ndarray,
    alpha: float = ALPHA,
    m: int = M_SIMULATIONS
) -> float:
    max_powers = []
    n = len(residuals)
    block_len = max(3, int(np.sqrt(n)))

    for _ in range(m):
        n_blocks = int(np.ceil(n / block_len))
        block_starts = np.random.randint(0, n - block_len + 1, n_blocks)

        boot_resid = []
        for start in block_starts:
            boot_resid.extend(residuals[start:start + block_len])
        boot_resid = np.array(boot_resid[:n])

        try:
            power = LombScargle(t, boot_resid, normalization='standard').power(freqs)
            max_powers.append(np.max(power))
        except:
            boot_resid_simple = np.random.choice(residuals, size=n, replace=True)
            power = LombScargle(t, boot_resid_simple, normalization='standard').power(freqs)
            max_powers.append(np.max(power))

    return np.percentile(max_powers, (1 - alpha) * 100)


def get_spectral_features(t_points: np.ndarray, periods: list) -> pd.DataFrame:
    feats = {}
    for p in periods:
        w = 2 * np.pi / p
        feats[f'sin_{int(p)}'] = np.sin(w * t_points)
        feats[f'cos_{int(p)}'] = np.cos(w * t_points)
    return pd.DataFrame(feats)


def prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() for c in df.columns]
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()

    df['Returns'] = safe_log(df['Close']) - safe_log(df['Close'].shift(1))
    df['Ret_sq'] = df['Returns'] ** 2

    rvd = df['Ret_sq'].groupby(df.index.date).sum()
    rvd.index = pd.to_datetime(rvd.index)
    rvd = rvd[rvd > 0]

    reference_date = rvd.index.min()
    time_days = (rvd.index - reference_date).days.values

    rvw = rvd.rolling(window=5, min_periods=5).mean()
    rvm = rvd.rolling(window=22, min_periods=22).mean()

    result = pd.DataFrame({
        'rvd': rvd,
        'rvw': rvw,
        'rvm': rvm,
        'time_days': time_days,
        'Target_Actual_RV': rvd.shift(-1),
        'Target_Log_RV': safe_log(rvd).shift(-1)
    })

    return result.dropna()




class PeriodTracker:
    """
    Tracks period stability across consecutive detection windows.

    Requires exactly window_size (= PERSISTENCE_WINDOW = 3) consecutive
    detections whose coefficient of variation is below PERIOD_TOLERANCE before
    activating spectral augmentation. A single missing detection resets the
    history, enforcing strict structural reliability.
    """

    def __init__(self, window_size: int = PERSISTENCE_WINDOW,
                 tolerance: float = PERIOD_TOLERANCE):
        self.window_size = window_size
        self.tolerance = tolerance
        self.history = []

    def update(self, period: Optional[float]) -> Tuple[bool, Optional[float]]:
        if period is None:
            self.history = []
            return False, None

        self.history.append(period)

        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) >= self.window_size:
            recent = np.array(self.history[-self.window_size:])
            mean_period = np.mean(recent)
            cv = np.std(recent) / mean_period

            if cv < self.tolerance:
                return True, mean_period

        return False, None



def aggregate_weekly_rv(train_rvd: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
   
    values = train_rvd.values
    n = len(values)
    n_complete = (n // 5) * 5          # trim to multiple of 5
    values = values[n - n_complete:]   # keep newest n_complete days

    # shape: (n_weeks, 5)
    blocks = values.reshape(-1, 5)
    rv_weekly = blocks.sum(axis=1)     # sum within each week

    # time coordinate in weeks (0, 1, 2, …)
    t_weeks = np.arange(len(rv_weekly), dtype=float)

    return t_weeks, rv_weekly


def process_single_file(filename: str, input_folder: str) -> dict:
    file_path = os.path.join(input_folder, filename)

    try:
        data = prepare_data(file_path)
        test_start = pd.to_datetime('2023-01-01')
        idx_start = data.index.searchsorted(test_start)

        results = []

        
        tracker_lshar  = PeriodTracker()
        tracker_resid  = PeriodTracker()
        tracker_weekly = PeriodTracker()   

        
        fap_lshar  = None;  prev_std_lshar  = None
        fap_resid  = None;  prev_std_resid  = None
        fap_weekly = None;  prev_std_weekly = None   

        for i in range(idx_start, len(data)):
            if i < LOOKBACK_DAILY:
                continue

            train = data.iloc[i - LOOKBACK_DAILY: i]
            test_row = data.iloc[[i]]
            actual_rv = test_row['Target_Actual_RV'].iloc[0]

            t_train = train['time_days'].values
            t_test  = test_row['time_days'].values[0]

            freqs = get_data_driven_freqs(t_train, n_freqs=100)

            
            X_train = sm.add_constant(train[['rvd', 'rvw', 'rvm']])
            X_train.columns = ['const', 'rvd', 'rvw', 'rvm']

            X_train_log = X_train.copy()
            X_train_log[['rvd', 'rvw', 'rvm']] = safe_log(X_train[['rvd', 'rvw', 'rvm']])

            har_model = sm.OLS(train['Target_Log_RV'].values, X_train_log).fit()

            X_test = sm.add_constant(test_row[['rvd', 'rvw', 'rvm']], has_constant='add')
            X_test.columns = ['const', 'rvd', 'rvw', 'rvm']
            X_test_log = X_test.copy()
            X_test_log[['rvd', 'rvw', 'rvm']] = safe_log(X_test[['rvd', 'rvw', 'rvm']])

            pred_har_log = har_model.predict(X_test_log).iloc[0]
            pred_har     = np.exp(pred_har_log + har_model.mse_resid / 2)

            
            log_rv_current = safe_log(train['rvd']).values

            if fap_lshar is None or prev_std_lshar is None or \
               abs(log_rv_current.std() - prev_std_lshar) > 0.2 * prev_std_lshar or i % 30 == 0:
                fap_lshar       = get_bootstrap_threshold(t_train, log_rv_current, freqs)
                prev_std_lshar  = log_rv_current.std()

            p_lshar   = LombScargle(t_train, log_rv_current, normalization='standard').power(freqs)
            peaks_a, _ = find_peaks(p_lshar, height=fap_lshar)

            detected_p_lshar = None
            if len(peaks_a) > 0:
                detected_p_lshar = 1.0 / freqs[peaks_a[np.argmax(p_lshar[peaks_a])]]

            active_lshar, stable_p_lshar = tracker_lshar.update(detected_p_lshar)
            pred_lshar = pred_har

            t_train_next = t_train + 1
            t_test_next  = t_test + 1

            if active_lshar and stable_p_lshar is not None:
                ls_train   = get_spectral_features(t_train_next, [stable_p_lshar])
                X_lshar    = pd.concat([
                    pd.DataFrame(X_train_log.values, columns=X_train_log.columns),
                    ls_train.reset_index(drop=True)
                ], axis=1)

                m_lshar    = sm.OLS(train['Target_Log_RV'].values, X_lshar).fit()

                ls_test    = get_spectral_features(np.array([t_test_next]), [stable_p_lshar])
                X_test_lshar = pd.concat([
                    pd.DataFrame(X_test_log.values, columns=X_test_log.columns),
                    ls_test.reset_index(drop=True)
                ], axis=1)
                X_test_lshar.columns = X_lshar.columns

                pred_lshar_log = m_lshar.predict(X_test_lshar).iloc[0]
                pred_lshar     = np.exp(pred_lshar_log + m_lshar.mse_resid / 2)

            
            # ResidLS: residuals computed against the same next-day log-RV
            # target the HAR model was trained on, so actual and fitted values
            # are time-aligned. t_train_next used for both the bootstrap and
            # the periodogram to match the regression target horizon.
            resids = train['Target_Log_RV'].values - har_model.predict(X_train_log)

            if fap_resid is None or prev_std_resid is None or \
               abs(resids.std() - prev_std_resid) > 0.2 * prev_std_resid or i % 30 == 0:
                fap_resid      = get_bootstrap_threshold(t_train_next, resids, freqs)
                prev_std_resid = resids.std()

            p_resid   = LombScargle(t_train_next, resids, normalization='standard').power(freqs)
            peaks_b, _ = find_peaks(p_resid, height=fap_resid)

            detected_p_resid = None
            if len(peaks_b) > 0:
                detected_p_resid = 1.0 / freqs[peaks_b[np.argmax(p_resid[peaks_b])]]

            active_resid, stable_p_resid = tracker_resid.update(detected_p_resid)
            pred_resid = pred_har

            if active_resid and stable_p_resid is not None:
                ls_train_r = get_spectral_features(t_train, [stable_p_resid])
                X_resid    = pd.concat([
                    pd.DataFrame(X_train_log.values, columns=X_train_log.columns),
                    ls_train_r.reset_index(drop=True)
                ], axis=1)

                m_resid    = sm.OLS(train['Target_Log_RV'].values, X_resid).fit()

                ls_test_r  = get_spectral_features(np.array([t_test]), [stable_p_resid])
                X_test_resid = pd.concat([
                    pd.DataFrame(X_test_log.values, columns=X_test_log.columns),
                    ls_test_r.reset_index(drop=True)
                ], axis=1)
                X_test_resid.columns = X_resid.columns

                pred_resid_log = m_resid.predict(X_test_resid).iloc[0]
                pred_resid     = np.exp(pred_resid_log + m_resid.mse_resid / 2)

        
            pred_weekly = pred_har   
            detected_p_weekly = None

            
            weekly_start = max(0, i - LOOKBACK_WEEKLY_LS)
            train_long   = data.iloc[weekly_start: i]

            
            if len(train_long) >= 100:
                t_weeks, rv_weekly = aggregate_weekly_rv(train_long['rvd'])
                log_rv_weekly      = safe_log(rv_weekly)

                
                freqs_w = get_data_driven_freqs(t_weeks, n_freqs=100)

                
                if fap_weekly is None or prev_std_weekly is None or \
                   abs(log_rv_weekly.std() - prev_std_weekly) > 0.2 * prev_std_weekly or i % 30 == 0:
                    fap_weekly      = get_bootstrap_threshold(t_weeks, log_rv_weekly, freqs_w)
                    prev_std_weekly = log_rv_weekly.std()

                p_weekly   = LombScargle(t_weeks, log_rv_weekly, normalization='standard').power(freqs_w)
                peaks_w, _ = find_peaks(p_weekly, height=fap_weekly)

                if len(peaks_w) > 0:
                    
                    best_period_weeks  = 1.0 / freqs_w[peaks_w[np.argmax(p_weekly[peaks_w])]]
                    detected_p_weekly  = best_period_weeks * 5.0   # weeks → trading days

            active_weekly, stable_p_weekly = tracker_weekly.update(detected_p_weekly)

            if active_weekly and stable_p_weekly is not None:
                ls_train_w = get_spectral_features(t_train_next, [stable_p_weekly])
                X_weekly   = pd.concat([
                    pd.DataFrame(X_train_log.values, columns=X_train_log.columns),
                    ls_train_w.reset_index(drop=True)
                ], axis=1)

                m_weekly   = sm.OLS(train['Target_Log_RV'].values, X_weekly).fit()

                ls_test_w  = get_spectral_features(np.array([t_test_next]), [stable_p_weekly])
                X_test_w   = pd.concat([
                    pd.DataFrame(X_test_log.values, columns=X_test_log.columns),
                    ls_test_w.reset_index(drop=True)
                ], axis=1)
                X_test_w.columns = X_weekly.columns

                pred_weekly_log = m_weekly.predict(X_test_w).iloc[0]
                pred_weekly     = np.exp(pred_weekly_log + m_weekly.mse_resid / 2)

            
            results.append({
                'Date':          data.index[i],
                'Actual':        actual_rv,
                'HAR':           pred_har,
                
                'LSHAR':         pred_lshar,
                'LSHAR_Active':  active_lshar,
                'LSHAR_Period':  stable_p_lshar if active_lshar else np.nan,
                
                'ResidLS':       pred_resid,
                'ResidLS_Active': active_resid,
                'ResidLS_Period': stable_p_resid if active_resid else np.nan,
                
                'WeeklyLS':          pred_weekly,
                'WeeklyLS_Active':   active_weekly,
                'WeeklyLS_Period':   stable_p_weekly if active_weekly else np.nan,
            })

        
        df_res = pd.DataFrame(results)

        lshar_count  = df_res['LSHAR_Active'].sum()
        resid_count  = df_res['ResidLS_Active'].sum()
        weekly_count = df_res['WeeklyLS_Active'].sum()

        
        if lshar_count > 0:
            mask = df_res['LSHAR_Active']
            lshar_qlike     = qlike_loss(df_res.loc[mask, 'Actual'], df_res.loc[mask, 'LSHAR']).mean()
            har_on_lshar    = qlike_loss(df_res.loc[mask, 'Actual'], df_res.loc[mask, 'HAR']).mean()
            lshar_improvement = (har_on_lshar - lshar_qlike) / abs(har_on_lshar) * 100
        else:
            lshar_qlike = har_on_lshar = np.nan
            lshar_improvement = 0

        
        if resid_count > 0:
            mask = df_res['ResidLS_Active']
            resid_qlike     = qlike_loss(df_res.loc[mask, 'Actual'], df_res.loc[mask, 'ResidLS']).mean()
            har_on_resid    = qlike_loss(df_res.loc[mask, 'Actual'], df_res.loc[mask, 'HAR']).mean()
            resid_improvement = (har_on_resid - resid_qlike) / abs(har_on_resid) * 100
        else:
            resid_qlike = har_on_resid = np.nan
            resid_improvement = 0

        
        if weekly_count > 0:
            mask = df_res['WeeklyLS_Active']
            weekly_qlike      = qlike_loss(df_res.loc[mask, 'Actual'], df_res.loc[mask, 'WeeklyLS']).mean()
            har_on_weekly     = qlike_loss(df_res.loc[mask, 'Actual'], df_res.loc[mask, 'HAR']).mean()
            weekly_improvement = (har_on_weekly - weekly_qlike) / abs(har_on_weekly) * 100
        else:
            weekly_qlike = har_on_weekly = np.nan
            weekly_improvement = 0

        
        har_global    = qlike_loss(df_res['Actual'], df_res['HAR']).mean()
        lshar_global  = qlike_loss(df_res['Actual'], df_res['LSHAR']).mean()
        resid_global  = qlike_loss(df_res['Actual'], df_res['ResidLS']).mean()
        weekly_global = qlike_loss(df_res['Actual'], df_res['WeeklyLS']).mean()

        print(f"{filename} | LSHAR: {lshar_count}d ({lshar_improvement:+.2f}%) | "
              f"ResidLS: {resid_count}d ({resid_improvement:+.2f}%) | "
              f"WeeklyLS: {weekly_count}d ({weekly_improvement:+.2f}%)")

        return {
            'Stock':               filename,
            'Total_Test_Days':     len(df_res),
            
            'HAR_QLIKE_Global':    har_global,
            'LSHAR_QLIKE_Global':  lshar_global,
            'ResidLS_QLIKE_Global': resid_global,
            'WeeklyLS_QLIKE_Global': weekly_global,
            
            'LSHAR_Active_Days':       lshar_count,
            'LSHAR_Active_Pct':        lshar_count / len(df_res) * 100,
            'HAR_on_LSHAR_Days':       har_on_lshar,
            'LSHAR_on_Active_Days':    lshar_qlike,
            'LSHAR_Improvement_Pct':   lshar_improvement,
            
            'ResidLS_Active_Days':     resid_count,
            'ResidLS_Active_Pct':      resid_count / len(df_res) * 100,
            'HAR_on_ResidLS_Days':     har_on_resid,
            'ResidLS_on_Active_Days':  resid_qlike,
            'ResidLS_Improvement_Pct': resid_improvement,
            
            'WeeklyLS_Active_Days':      weekly_count,
            'WeeklyLS_Active_Pct':       weekly_count / len(df_res) * 100,
            'HAR_on_WeeklyLS_Days':      har_on_weekly,
            'WeeklyLS_on_Active_Days':   weekly_qlike,
            'WeeklyLS_Improvement_Pct':  weekly_improvement,
        }

    except Exception as e:
        print(f"✗ ERROR in {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'Stock': filename, 'Error': str(e)}



def main():
    """Run full analysis pipeline."""
    parser = argparse.ArgumentParser(description="LSHAR Model Analysis")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Path to folder containing CSV files. "
                             "If not provided, a folder picker dialog will open.")
    args = parser.parse_args()

    if args.data_dir:
        path = args.data_dir
    else:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select Data Folder")
        if not path:
            return

    files = [f for f in os.listdir(path) if f.endswith('.csv')]

    print("=" * 80)
    print("LOMB-SCARGLE VOLATILITY FORECASTING ANALYSIS  (v5 — WeeklyLS track)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Significance level:          {ALPHA}")
    print(f"  - Bootstrap samples:           {M_SIMULATIONS}")
    print(f"  - Persistence window:          {PERSISTENCE_WINDOW}")
    print(f"  - Daily lookback (HAR/LS):     {LOOKBACK_DAILY} days")
    print(f"  - Weekly LS lookback:          {LOOKBACK_WEEKLY_LS} days (~{LOOKBACK_WEEKLY_LS//5} weeks)")
    print(f"\nProcessing {len(files)} stocks using {os.cpu_count()} cores...")
    print("=" * 80 + "\n")

    with ProcessPoolExecutor() as ex:
        final_results = list(tqdm(
            ex.map(partial(process_single_file, input_folder=path), files),
            total=len(files), desc="Processing stocks", unit="stock",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} stocks  [{elapsed}<{remaining}]"
        ))

    summary = pd.DataFrame([r for r in final_results if 'Error' not in r])
    errors  = pd.DataFrame([r for r in final_results if 'Error' in r])

    summary.to_csv(os.path.join(path, f"LSHAR_Publication_Results{PERSISTENCE_WINDOW}.csv"), index=False)
    if len(errors) > 0:
        errors.to_csv(os.path.join(path, f"Processing_Errors{PERSISTENCE_WINDOW}.csv"), index=False)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {len(summary)} stocks")
    print(f"Errors: {len(errors)}")

    if len(summary) > 0:
        def _track_summary(label, active_col, improv_col):
            s = summary[summary[active_col] > 0]
            print(f"\n{'-'*80}")
            print(label)
            print(f"{'-'*80}")
            print(f"  Mean active days:    {summary[active_col].mean():.1f} ({summary[active_col.replace('_Days','_Pct')].mean():.1f}%)")
            print(f"  Mean improvement:    {s[improv_col].mean():+.3f}%")
            print(f"  Median improvement:  {s[improv_col].median():+.3f}%")
            print(f"  Positive stocks:     {(s[improv_col] > 0).sum()}/{len(s)}")

        _track_summary("LSHAR  (Direct Spectral Augmentation — daily LS)",
                       'LSHAR_Active_Days',   'LSHAR_Improvement_Pct')
        _track_summary("RESIDLS (Residual Spectral Analysis — daily LS)",
                       'ResidLS_Active_Days', 'ResidLS_Improvement_Pct')
        _track_summary("WEEKLYLS (Weekly Aggregated LS — 2-year window)",
                       'WeeklyLS_Active_Days', 'WeeklyLS_Improvement_Pct')

    print("\n" + "=" * 80)
    print(f"Results saved to: LSHAR_Publication_Results.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
