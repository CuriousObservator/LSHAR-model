#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:21:30 2026

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

warnings.filterwarnings("ignore")

# ==========================
# CONFIGURATION
# ==========================
ALPHA = 0.05  # Significance level for periodogram peaks
M_SIMULATIONS = 1000  # Monte Carlo simulations (increased for robustness)
LOOKBACK_DAILY = 252  # One year of trading days
PERSISTENCE_WINDOW = 3  # Require period stability over 3 windows
PERIOD_TOLERANCE = 0.15  # 15% tolerance for period consistency

# ==========================
# UTILITY FUNCTIONS
# ==========================

def safe_log(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:

    return np.log(np.maximum(x, floor))

def qlike_loss(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    forecast = np.maximum(forecast, 1e-10)
    return np.log(forecast) + (actual / forecast)




def get_data_driven_freqs(t: np.ndarray, n_freqs: int = 100) -> np.ndarray:
    T = t.max() - t.min()  # Total time span
    N = len(t)  # Number of observations
    
    # Minimum frequency: one cycle over entire span
    f_min = 1.0 / T
    
    # Pseudo-Nyquist for irregular sampling
    # Conservative: half the average sampling rate
    avg_sampling_rate = N / T
    f_max = 0.5 * avg_sampling_rate
    
    # Create log-spaced grid (better resolution at low frequencies)
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
    
    # Block length: ~sqrt(n) for balance between independence and structure
    block_len = max(3, int(np.sqrt(n)))
    
    for _ in range(m):
        # Block bootstrap to preserve autocorrelation
        n_blocks = int(np.ceil(n / block_len))
        block_starts = np.random.randint(0, n - block_len + 1, n_blocks)
        
        boot_resid = []
        for start in block_starts:
            boot_resid.extend(residuals[start:start + block_len])
        boot_resid = np.array(boot_resid[:n])
        
        # Compute periodogram power
        try:
            power = LombScargle(t, boot_resid, normalization='standard').power(freqs)
            max_powers.append(np.max(power))
        except:
            # Fallback to simple resampling if LS fails
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

# ==========================
# DATA PREPARATION
# ==========================

def prepare_data(file_path: str) -> pd.DataFrame:
    # Load and standardize
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() for c in df.columns]
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Compute returns
    df['Returns'] = safe_log(df['Close']) - safe_log(df['Close'].shift(1))
    df['Ret_sq'] = df['Returns'] ** 2
    
    # Aggregate to daily RV WITHOUT creating uniform grid
    rvd = df['Ret_sq'].groupby(df.index.date).sum()
    rvd.index = pd.to_datetime(rvd.index)
    rvd = rvd[rvd > 0]  # Keep only trading days
    
    # Calendar time (preserves weekend/holiday gaps)
    reference_date = rvd.index.min()
    time_days = (rvd.index - reference_date).days.values
    
    # HAR components (on trading days)
    rvw = rvd.rolling(window=5, min_periods=5).mean()  # Weekly
    rvm = rvd.rolling(window=22, min_periods=22).mean()  # Monthly
    
    # Construct feature set
    result = pd.DataFrame({
        'rvd': rvd,
        'rvw': rvw,
        'rvm': rvm,
        'time_days': time_days,
        'Target_Actual_RV': rvd.shift(-1),
        'Target_Log_RV': safe_log(rvd).shift(-1)
    })
    
    return result.dropna()

# ==========================
# PERIOD DETECTION WITH PERSISTENCE FILTER
# ==========================

class PeriodTracker:
    
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
        
        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Check for stability
        if len(self.history) >= 2:
            recent = np.array(self.history[-min(self.window_size, len(self.history)):])
            mean_period = np.mean(recent)
            cv = np.std(recent) / mean_period  # Coefficient of variation
            
            if cv < self.tolerance:
                return True, mean_period
        
        return False, None

# ==========================
# CORE FORECASTING WORKER
# ==========================

def process_single_file(filename: str, input_folder: str) -> dict:
    file_path = os.path.join(input_folder, filename)
    
    try:
        data = prepare_data(file_path)
        test_start = pd.to_datetime('2023-01-01')
        idx_start = data.index.searchsorted(test_start)
        
        results = []
        
        # Initialize trackers
        tracker_lshar = PeriodTracker()
        tracker_resid = PeriodTracker()
        
        # Cache for adaptive thresholds
        fap_lshar = None
        fap_resid = None
        prev_std_lshar = None
        prev_std_resid = None

        for i in range(idx_start, len(data)):
            if i < LOOKBACK_DAILY:
                continue
            
            train = data.iloc[i - LOOKBACK_DAILY: i]
            test_row = data.iloc[[i]]
            actual_rv = test_row['Target_Actual_RV'].iloc[0]
            
            # Irregular time coordinates
            t_train = train['time_days'].values
            t_test = test_row['time_days'].values[0]
            
            # Data-driven frequency grid
            freqs = get_data_driven_freqs(t_train, n_freqs=100)
            
            # ============================================
            # 1. HAR BASELINE
            # ============================================
            X_train = sm.add_constant(train[['rvd', 'rvw', 'rvm']])
            X_train.columns = ['const', 'rvd', 'rvw', 'rvm']
            
            # Log-transform HAR components
            X_train_log = X_train.copy()
            X_train_log[['rvd', 'rvw', 'rvm']] = safe_log(X_train[['rvd', 'rvw', 'rvm']])
            
            har_model = sm.OLS(train['Target_Log_RV'].values, X_train_log).fit()
            
            # Test prediction
            X_test = sm.add_constant(test_row[['rvd', 'rvw', 'rvm']], has_constant='add')
            X_test.columns = ['const', 'rvd', 'rvw', 'rvm']
            X_test_log = X_test.copy()
            X_test_log[['rvd', 'rvw', 'rvm']] = safe_log(X_test[['rvd', 'rvw', 'rvm']])
            
            pred_har_log = har_model.predict(X_test_log).iloc[0]
            pred_har = np.exp(pred_har_log + har_model.mse_resid / 2)
            
            # ============================================
            # 2. TRACK A: LSHAR
            # ============================================
            log_rv = train['Target_Log_RV'].values
            
            # Adaptive threshold (recompute if volatility regime shifts)
            if fap_lshar is None or prev_std_lshar is None or \
               abs(log_rv.std() - prev_std_lshar) > 0.2 * prev_std_lshar or i % 30 == 0:
                fap_lshar = get_bootstrap_threshold(t_train, log_rv, freqs)
                prev_std_lshar = log_rv.std()
            
            # Periodogram
            p_lshar = LombScargle(t_train, log_rv, normalization='standard').power(freqs)
            peaks_a, _ = find_peaks(p_lshar, height=fap_lshar)
            
            # Detect strongest period
            detected_p_lshar = None
            if len(peaks_a) > 0:
                detected_p_lshar = 1.0 / freqs[peaks_a[np.argmax(p_lshar[peaks_a])]]
            
            # Check persistence
            active_lshar, stable_p_lshar = tracker_lshar.update(detected_p_lshar)
            pred_lshar = pred_har  # Default
            
            if active_lshar and stable_p_lshar is not None:
                # Augment HAR with spectral features
                ls_train = get_spectral_features(t_train, [stable_p_lshar])
                X_lshar = pd.concat([
                    pd.DataFrame(X_train_log.values, columns=X_train_log.columns),
                    ls_train.reset_index(drop=True)
                ], axis=1)
                
                m_lshar = sm.OLS(log_rv, X_lshar).fit()
                
                ls_test = get_spectral_features(np.array([t_test]), [stable_p_lshar])
                X_test_lshar = pd.concat([
                    pd.DataFrame(X_test_log.values, columns=X_test_log.columns),
                    ls_test.reset_index(drop=True)
                ], axis=1)
                X_test_lshar.columns = X_lshar.columns
                
                pred_lshar_log = m_lshar.predict(X_test_lshar).iloc[0]
                pred_lshar = np.exp(pred_lshar_log + m_lshar.mse_resid / 2)
            
            # ============================================
            # 3. TRACK B: ResidLS
            # ============================================
            resids = log_rv - har_model.predict(X_train_log)
            
            # Adaptive threshold
            if fap_resid is None or prev_std_resid is None or \
               abs(resids.std() - prev_std_resid) > 0.2 * prev_std_resid or i % 30 == 0:
                fap_resid = get_bootstrap_threshold(t_train, resids, freqs)
                prev_std_resid = resids.std()
            
            # Periodogram on residuals
            p_resid = LombScargle(t_train, resids, normalization='standard').power(freqs)
            peaks_b, _ = find_peaks(p_resid, height=fap_resid)
            
            detected_p_resid = None
            if len(peaks_b) > 0:
                detected_p_resid = 1.0 / freqs[peaks_b[np.argmax(p_resid[peaks_b])]]
            
            active_resid, stable_p_resid = tracker_resid.update(detected_p_resid)
            pred_resid = pred_har  # Default
            
            if active_resid and stable_p_resid is not None:
                ls_train_r = get_spectral_features(t_train, [stable_p_resid])
                X_resid = pd.concat([
                    pd.DataFrame(X_train_log.values, columns=X_train_log.columns),
                    ls_train_r.reset_index(drop=True)
                ], axis=1)
                
                m_resid = sm.OLS(log_rv, X_resid).fit()
                
                ls_test_r = get_spectral_features(np.array([t_test]), [stable_p_resid])
                X_test_resid = pd.concat([
                    pd.DataFrame(X_test_log.values, columns=X_test_log.columns),
                    ls_test_r.reset_index(drop=True)
                ], axis=1)
                X_test_resid.columns = X_resid.columns
                
                pred_resid_log = m_resid.predict(X_test_resid).iloc[0]
                pred_resid = np.exp(pred_resid_log + m_resid.mse_resid / 2)
            
            results.append({
                'Date': data.index[i],
                'Actual': actual_rv,
                'HAR': pred_har,
                'LSHAR': pred_lshar,
                'LSHAR_Active': active_lshar,
                'LSHAR_Period': stable_p_lshar if active_lshar else np.nan,
                'ResidLS': pred_resid,
                'ResidLS_Active': active_resid,
                'ResidLS_Period': stable_p_resid if active_resid else np.nan
            })
        
        # ============================================
        # PERFORMANCE EVALUATION
        # ============================================
        df_res = pd.DataFrame(results)
        
        # Active day counts
        lshar_count = df_res['LSHAR_Active'].sum()
        resid_count = df_res['ResidLS_Active'].sum()
        
        # LSHAR vs HAR (on same days)
        if lshar_count > 0:
            mask = df_res['LSHAR_Active']
            lshar_qlike = qlike_loss(df_res.loc[mask, 'Actual'], 
                                     df_res.loc[mask, 'LSHAR']).mean()
            har_on_lshar = qlike_loss(df_res.loc[mask, 'Actual'], 
                                      df_res.loc[mask, 'HAR']).mean()
            lshar_improvement = ((har_on_lshar - lshar_qlike) / har_on_lshar * 100)
        else:
            lshar_qlike = np.nan
            har_on_lshar = np.nan
            lshar_improvement = 0
        
        # ResidLS vs HAR (on same days)
        if resid_count > 0:
            mask = df_res['ResidLS_Active']
            resid_qlike = qlike_loss(df_res.loc[mask, 'Actual'], 
                                     df_res.loc[mask, 'ResidLS']).mean()
            har_on_resid = qlike_loss(df_res.loc[mask, 'Actual'], 
                                      df_res.loc[mask, 'HAR']).mean()
            resid_improvement = ((har_on_resid - resid_qlike) / har_on_resid * 100)
        else:
            resid_qlike = np.nan
            har_on_resid = np.nan
            resid_improvement = 0
        
        # Global metrics
        har_global = qlike_loss(df_res['Actual'], df_res['HAR']).mean()
        lshar_global = qlike_loss(df_res['Actual'], df_res['LSHAR']).mean()
        resid_global = qlike_loss(df_res['Actual'], df_res['ResidLS']).mean()
        
        # Save detailed results
        # output_path = os.path.join(input_folder, f"details_{filename}")
        # df_res.to_csv(output_path, index=False)
        
        print(f"{filename} | LSHAR: {lshar_count} days ({lshar_improvement:+.2f}%) | "
              f"ResidLS: {resid_count} days ({resid_improvement:+.2f}%)")
        
        return {
            'Stock': filename,
            'Total_Test_Days': len(df_res),
            'HAR_QLIKE_Global': har_global,
            'LSHAR_QLIKE_Global': lshar_global,
            'ResidLS_QLIKE_Global': resid_global,
            'LSHAR_Active_Days': lshar_count,
            'LSHAR_Active_Pct': (lshar_count / len(df_res) * 100),
            'HAR_on_LSHAR_Days': har_on_lshar,
            'LSHAR_on_Active_Days': lshar_qlike,
            'LSHAR_Improvement_Pct': lshar_improvement,
            'ResidLS_Active_Days': resid_count,
            'ResidLS_Active_Pct': (resid_count / len(df_res) * 100),
            'HAR_on_ResidLS_Days': har_on_resid,
            'ResidLS_on_Active_Days': resid_qlike,
            'ResidLS_Improvement_Pct': resid_improvement
        }
    
    except Exception as e:
        print(f"✗ ERROR in {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'Stock': filename, 'Error': str(e)}

# ==========================
# MAIN EXECUTION
# ==========================

def main():
    """Run full analysis pipeline."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title="Select Data Folder")
    if not path:
        return
    
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    print("="*80)
    print("LOMB-SCARGLE VOLATILITY FORECASTING ANALYSIS")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Significance level: {ALPHA}")
    print(f"  - Bootstrap samples: {M_SIMULATIONS}")
    print(f"  - Persistence window: {PERSISTENCE_WINDOW}")
    print(f"  - Lookback period: {LOOKBACK_DAILY} days")
    print(f"\nProcessing {len(files)} stocks using {os.cpu_count()} cores...")
    print("="*80 + "\n")
    
    with ProcessPoolExecutor() as ex:
        final_results = list(ex.map(partial(process_single_file, input_folder=path), files))
    
    summary = pd.DataFrame([r for r in final_results if 'Error' not in r])
    errors = pd.DataFrame([r for r in final_results if 'Error' in r])
    
    # Save results
    summary.to_csv(os.path.join(path, "LSHAR_Publication_Results.csv"), index=False)
    if len(errors) > 0:
        errors.to_csv(os.path.join(path, "Processing_Errors.csv"), index=False)
    
    # Summary statistics
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Successfully processed: {len(summary)} stocks")
    print(f"Errors: {len(errors)}")
    
    if len(summary) > 0:
        print("\n" + "-"*80)
        print("LSHAR TRACK (Direct Spectral Augmentation)")
        print("-"*80)
        print(f"Mean active days: {summary['LSHAR_Active_Days'].mean():.1f} "
              f"({summary['LSHAR_Active_Pct'].mean():.1f}%)")
        print(f"Mean improvement: {summary['LSHAR_Improvement_Pct'].mean():+.3f}%")
        print(f"Median improvement: {summary['LSHAR_Improvement_Pct'].median():+.3f}%")
        print(f"Positive improvements: {(summary['LSHAR_Improvement_Pct'] > 0).sum()}/{len(summary)} stocks")
        
        print("\n" + "-"*80)
        print("RESIDLS TRACK (Residual Spectral Analysis)")
        print("-"*80)
        print(f"Mean active days: {summary['ResidLS_Active_Days'].mean():.1f} "
              f"({summary['ResidLS_Active_Pct'].mean():.1f}%)")
        print(f"Mean improvement: {summary['ResidLS_Improvement_Pct'].mean():+.3f}%")
        print(f"Median improvement: {summary['ResidLS_Improvement_Pct'].median():+.3f}%")
        print(f"Positive improvements: {(summary['ResidLS_Improvement_Pct'] > 0).sum()}/{len(summary)} stocks")
    
    print("\n" + "="*80)
    print(f"Results saved to: LSHAR_Publication_Results.csv")
    print("="*80)

if __name__ == "__main__":
    main()
