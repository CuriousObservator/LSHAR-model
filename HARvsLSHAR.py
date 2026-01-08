#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:51:10 2025

@author: rbarcrosspbar
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import statsmodels.api as sm
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# from datetime import timedelta,datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


alpha = 0.1
def qlike_loss(actual, forecast):
    return np.log(actual/forecast) + actual / forecast -1

def rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast)**2))

def calc_daily_rv(data):
    start_time = pd.to_datetime('09:15:00').time() or pd.to_datetime('09:20:00').time()
    end_time = pd.to_datetime('15:30:00').time()

    in_session_data = data.between_time(start_time, end_time, inclusive='both')
    daily_rv_series = in_session_data['Ret_sq'].resample('D').sum()
    daily_rv_series = daily_rv_series[daily_rv_series > 0].rename('RV_t')
    return daily_rv_series

##### This section does the manual monte-carlo simulation for the FAP calculation

# def mc_fap(t, y, freq, m=1000):
#     max_powers = []
#     y_std = np.std(y)
#     y_mean = np.mean(y)
#     for i in range(m):
#         y_sim = y_mean + np.random.normal(0,y_std,len(y))
#         model_sim = LombScargle(t, y_sim)
#         power_sim = model_sim.power(freq, normalization="standard")
#         max_powers.append(np.max(power_sim))
#     max_powers.sort()
#     fap_index = int((1-alpha)*m)
#     fap_threshold = max_powers[fap_index]
#     return fap_threshold

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
            executor.submit(_single_max_power_sim, t, y_mean, y_std, freq, len_y)
            for _ in range(m)
        ]
        
        for future in as_completed(futures):
            max_powers.append(future.result())

    max_powers.sort()
    
    fap_index = int(np.floor((1 - alpha) * m))
    fap_index = max(0, min(fap_index, m - 1)) 
    
    fap_threshold = max_powers[fap_index]
    
    return fap_threshold

####### We are analytically calculating the FAP

def LS_features(t_index, unq_periods):
    time_diffs = t_index - t_index[0]
    t = pd.to_timedelta(time_diffs).total_seconds().values / 86400 
    ls_features = pd.DataFrame(index=t_index)
    for n in unq_periods:
        f = 1/n
        ls_features[f'Sine_{n:.2f}'] = np.sin(2*np.pi*f*t)
        ls_features[f'Cos_{n:.2f}'] = np.cos(2*np.pi*f*t)
    return ls_features

def LS_features_weekly(t_index, unq_periods):
    time_diffs = t_index - t_index[0]
    t = pd.to_timedelta(time_diffs).total_seconds().values / 604800 
    ls_features = pd.DataFrame(index=t_index)
    for n in unq_periods:
        f = 1/n
        ls_features[f'Sine_{n:.2f}'] = np.sin(2*np.pi*f*t)
        ls_features[f'Cos_{n:.2f}'] = np.cos(2*np.pi*f*t)
    return ls_features

def select_robust_periods(freq, power, fap_threshold, frequency_distance, tolerance):
    peaks, properties = find_peaks(power, height=fap_threshold)
    sig_freqs = freq[peaks]
    sig_periods = 1.0 / sig_freqs
    sig_powers = power[peaks]

    results_df = pd.DataFrame({
        'Period_Days': sig_periods,
        'Power': sig_powers
    }).sort_values(by='Power', ascending=False).reset_index(drop=True)

    def filter_by_period_proximity(df, tolerance):
        selected_periods = []
        for _, row in df.iterrows():
            current_period = row['Period_Days']

            is_too_close = False
            for selected in selected_periods:
                if abs(current_period - selected) < tolerance:
                    is_too_close = True
                    break

            if not is_too_close:
                selected_periods.append(current_period)

        final_df = df[df['Period_Days'].isin(selected_periods)].reset_index(drop=True)
        return final_df

    final_periods_df = filter_by_period_proximity(results_df,tolerance=0.5)
    
    return final_periods_df['Period_Days'].tolist()


def create_LSHAR_features(Y, X_HAR, period):
    LS_F = LS_features(Y.index, period)
    X_LSHAR = pd.concat([X_HAR, LS_F], axis=1)
    X_LSHAR = sm.add_constant(X_LSHAR)
    Y = Y.reindex(X_LSHAR.index)
    return Y, X_LSHAR

def create_LSHAR_weekly_features(Y, X_HAR, period):
    LS_F = LS_features_weekly(Y.index, period)
    X_LSHAR = pd.concat([X_HAR, LS_F], axis=1)
    X_LSHAR = sm.add_constant(X_LSHAR)
    Y = Y.reindex(X_LSHAR.index)
    return Y, X_LSHAR

def get_ls_features(t_values, periods):
    """Generates Sine/Cosine features for absolute time t."""
    features = {}
    for n in periods:
        f = 1.0 / n
        features[f'Sine_{n:.2f}'] = np.sin(2 * np.pi * f * t_values)
        features[f'Cos_{n:.2f}'] = np.cos(2 * np.pi * f * t_values)
    return pd.DataFrame(features)



#%%
Tk().withdraw() 
file = askopenfilename() 

df = pd.read_csv(file)

df = df.set_index("Datetime").sort_index()

df.index = pd.to_datetime(df.index)

df["Returns"] = np.log(df["Close"]).diff()

df["Ret_sq"] = np.square(df["Returns"])

df.dropna(inplace=True)

print(df.head())
#%%
rvd = calc_daily_rv(df)
rvw = rvd.rolling(window = 5).mean()
rvm = rvd.rolling(window = 22).mean()

data_dict = {
    'rvd' : rvd,
    'rvw' : rvw,
    'rvm' : rvm
    }

Y = rvd.shift(-1)

rv_features = pd.DataFrame(data_dict)

rv_combined = pd.concat([Y.rename('Y'), rv_features], axis=1).dropna()

t_index = rv_combined.index

Y = np.log(rv_combined['Y'])
X = np.log(rv_combined[['rvd', 'rvw', 'rvm']])

Y.index = Y.index.date
X.index = X.index.date

#%%
Y_1 = rvd.shift(-1)
rv_combined_1 = pd.concat([Y_1.rename('Y_1'), rv_features], axis=1).dropna()
Y_1_log = np.log(rv_combined_1['Y_1'])
X_1_log = np.log(rv_combined_1[['rvd', 'rvw', 'rvm']])
RVd_full = rv_combined_1['rvd']
RVw_full = rv_combined_1['rvw']

Y_1_log.index = Y_1_log.index.date
X_1_log.index = X_1_log.index.date
RVd_full.index = RVd_full.index.date


oos_errors_HAR = {'QLIKE': {}, 'RMSE': {}}
oos_errors_Hybrid = {'QLIKE': {}, 'RMSE': {}}


oos_preds_1d_step_ahead = {} 
oos_errors = {'QLIKE': {}, 'RMSE': {}} 

lookback = 256
step_size = 1 
start_date_train = pd.to_datetime('2023-01-01').date() 
global_start = X_1_log.index[0]

#%%
try:
    start = X_1_log.index.get_loc(start_date_train) 
except KeyError:
    start = X_1_log.index.searchsorted(start_date_train)
if start < 0: start = 0

# fap_method = "manual"
fap_method = ""

while start + lookback < len(X_1_log)-1: 
    
    train_end_index = start + lookback
    
    x_har_train = X_1_log.iloc[start:train_end_index] 
    y_target_train = Y_1_log.iloc[start:train_end_index]
    y_ls = x_har_train['rvd'].values
    

    time_diffs = x_har_train.index - global_start
    t_ls = pd.to_timedelta(time_diffs).total_seconds().values / 86400
    freq_min = 1.0/lookback
    num_bins = int(5* lookback)
    freq = np.linspace(freq_min, 0.5*1.05, num_bins)
    model = LombScargle(t_ls, y_ls) 
    power = model.power(freq, normalization='standard')
    
    if fap_method == 'manual': fap_threshold = np.float64(mc_fap(t_ls, y_ls, freq, alpha))
    
    # print(fap_threshold)
    else:     fap_threshold = np.float64(model.false_alarm_level(alpha))

    period_distance = 1
    periods = select_robust_periods(freq, power, fap_threshold, 0.005, period_distance)
    
    
    
    oos_feature_index = train_end_index
    if oos_feature_index >= len(X_1_log): break
    
    x_oos_har = X_1_log.iloc[oos_feature_index]
    oos_target_date = Y_1_log.index[oos_feature_index]
    actual_rv = RVd_full.iloc[oos_feature_index + 1]
    
    oos_har_df = pd.DataFrame([x_oos_har], index=[oos_target_date])
    
    # HAR Model (Baseline) 
    try:
        x_har_train_const = sm.add_constant(x_har_train)
        y_har_train_aligned = y_target_train.reindex(x_har_train_const.index)
        har_results = sm.OLS(y_har_train_aligned, x_har_train_const).fit()

        x_oos_har_const = sm.add_constant(oos_har_df, has_constant='add')
        x_oos_har_const.columns = har_results.params.index # Align OOS columns
        
        har_log_pred = har_results.predict(x_oos_har_const).iloc[0]
        sigma2_har = har_results.mse_resid
        forecast_rv_har = np.exp(har_log_pred + sigma2_har / 2)
        

        oos_errors_HAR['QLIKE'][oos_target_date] = qlike_loss(actual_rv, forecast_rv_har)
        oos_errors_HAR['RMSE'][oos_target_date] = (actual_rv - forecast_rv_har)**2

        # LSHAR/Hybrid Model 
        if periods:
            
            y_lshar_train, x_lshar_train = create_LSHAR_features(y_target_train, x_har_train, periods)
            ols_results = sm.OLS(y_lshar_train, x_lshar_train).fit()

      
            oos_time = (oos_target_date - x_har_train.index[0]).total_seconds() / 86400
            oos_ls_features = pd.DataFrame(index=[oos_target_date])
            for n in periods:
                f = 1/n
                oos_ls_features[f'Sine_{n:.2f}'] = np.sin(2*np.pi*f*oos_time)
                oos_ls_features[f'Cos_{n:.2f}'] = np.cos(2*np.pi*f*oos_time)
            
            
            x_oos_har_no_const = oos_har_df.reset_index(drop=True)
            oos_ls_features_no_index = oos_ls_features.reset_index(drop=True)
            x_oos_lshar = pd.concat([x_oos_har_no_const, oos_ls_features_no_index], axis=1)
            x_oos_lshar = sm.add_constant(x_oos_lshar, has_constant='add') # Add constant back

            x_oos_lshar.columns = ols_results.params.index # Align columns to LSHAR params
            
            oos_log_pred = ols_results.predict(x_oos_lshar).iloc[0]
            sigma2_hybrid = ols_results.mse_resid
            forecast_rv_hybrid = np.exp(oos_log_pred + sigma2_hybrid / 2)
            

            oos_preds_1d_step_ahead[oos_target_date] = forecast_rv_hybrid
            oos_errors['QLIKE'][oos_target_date] = qlike_loss(actual_rv, forecast_rv_hybrid)
            oos_errors['RMSE'][oos_target_date] = (actual_rv - forecast_rv_hybrid)**2
            print(start)
            # plt.plot(freq, power)
            # # plt.title("Lomb-Scargle Periodogram")
            # plt.xlabel("Frequency-->")
            # plt.ylabel("Power-->")
            # plt.axhline(fap_threshold, color='red')
            # plt.text(0.4, 0.0625, 'fap_threshold', bbox=dict(facecolor='red', alpha=0.5))
            # plt.show()
            
        else:

            forecast_rv_hybrid = forecast_rv_har
            

        oos_errors_Hybrid['QLIKE'][oos_target_date] = qlike_loss(actual_rv, forecast_rv_hybrid)
        oos_errors_Hybrid['RMSE'][oos_target_date] = (actual_rv - forecast_rv_hybrid)**2
        
        start += step_size
        
    except Exception as e:
        print(f"Error for section-{start}: {e}")
        start += step_size
        continue

if oos_errors_HAR:
    final_qlike_har = np.mean(list(oos_errors_HAR['QLIKE'].values()))
    final_rmse_har = np.sqrt(np.mean(list(oos_errors_HAR['RMSE'].values())))
    total_har_preds = len(oos_errors_HAR['QLIKE'])

    final_qlike_hybrid = np.mean(list(oos_errors_Hybrid['QLIKE'].values()))
    final_rmse_hybrid = np.sqrt(np.mean(list(oos_errors_Hybrid['RMSE'].values())))
    
    
    print("Out-of-Sample Performance Comparison (Full Period)")
    print(f"Total Predictions (Full Period):** {total_har_preds}")
    print("---" * 15)
    
    print("HAR Baseline Model Performance")
    print(f"* Average QLIKE Loss: {final_qlike_har:.6f}")
    print(f"* Final RMSE: {final_rmse_har:.6f}")
    print("---" * 15)
    
    print("LSHAR/HAR Hybrid Model Performance")
    print(f"* Average QLIKE Loss: {final_qlike_hybrid:.6f}")
    print(f"* Final RMSE: {final_rmse_hybrid:.6f}")
    print("---" * 15)
    
    
    if oos_errors['QLIKE']:
        final_qlike_lshar_only = np.mean(list(oos_errors['QLIKE'].values()))
        final_rmse_lshar_only = np.sqrt(np.mean(list(oos_errors['RMSE'].values())))
        print("LSHAR-Only Model Performance (Diagnostic)")
        print(f"* **Predictions Count:** {len(oos_errors['QLIKE'])}")
        print(f"* Average QLIKE Loss: {final_qlike_lshar_only:.6f}")
        print(f"* Final RMSE: {final_rmse_lshar_only:.6f}")
        print("---" * 15)
else:
    print("No out-of-sample predictions were made.")
    
#%%

#we will now work with weekly data such that, the periodicities are much stabler

lookback = 256
step_size = 1 
start_date_train = pd.to_datetime('2023-01-01').date() 
lookback_weeks = 52
global_start = X_1_log.index[0]
try:
    start = X_1_log.index.get_loc(start_date_train) 
except KeyError:
    start = X_1_log.index.searchsorted(start_date_train)
if start < 0: start = 0

oos_errors = {'QLIKE': {}, 'RMSE': {}}

while start + lookback < len(X_1_log)-1: 
    
    train_end_index = start + lookback
    
    x_har_train = X_1_log.iloc[start:train_end_index] 
    y_target_train = Y_1_log.iloc[start:train_end_index]
    y_ls = x_har_train['rvw'].values
    

    time_diffs = x_har_train.index - global_start
    t_ls = pd.to_timedelta(time_diffs).total_seconds().values / 604800
    freq_min = 1.0/lookback_weeks
    num_bins = int(5* lookback_weeks)
    freq = np.linspace(freq_min, 1*1.05, num_bins)
    model = LombScargle(t_ls, y_ls) 
    power = model.power(freq, normalization='standard')
    
    # fap_threshold = np.float64(mc_fap(t_ls, y_ls, freq, alpha))
    fap_threshold = np.float64(model.false_alarm_level(alpha))
    period_distance = 1
    periods = select_robust_periods(freq, power, fap_threshold, 0.005, period_distance)
    
    # plt.plot(freq, power)
    # plt.text(0.4, np.max(power)-0.01, start+lookback)
    # plt.axhline(fap_threshold)
    # plt.show()
    
    
    
    oos_feature_index = train_end_index
    if oos_feature_index >= len(X_1_log): break
    
    x_oos_har = X_1_log.iloc[oos_feature_index]
    oos_target_date = Y_1_log.index[oos_feature_index]
    actual_rv = RVd_full.iloc[oos_feature_index + 1]
    oos_har_df = pd.DataFrame([x_oos_har], index=[oos_target_date])
    
    # HAR Model (Baseline) 
    try:
        x_har_train_const = sm.add_constant(x_har_train)
        y_har_train_aligned = y_target_train.reindex(x_har_train_const.index)
        har_results = sm.OLS(y_har_train_aligned, x_har_train_const).fit()

        x_oos_har_const = sm.add_constant(oos_har_df, has_constant='add')
        x_oos_har_const.columns = har_results.params.index # Align OOS columns
        
        har_log_pred = har_results.predict(x_oos_har_const).iloc[0]
        sigma2_har = har_results.mse_resid
        forecast_rv_har = np.exp(har_log_pred + sigma2_har / 2)
        

        oos_errors_HAR['QLIKE'][oos_target_date] = qlike_loss(actual_rv, forecast_rv_har)
        oos_errors_HAR['RMSE'][oos_target_date] = (actual_rv - forecast_rv_har)**2

        # LSHAR/Hybrid Model 
        if periods:
            
            y_lshar_train, x_lshar_train = create_LSHAR_weekly_features(y_target_train, x_har_train, periods)
            ols_results = sm.OLS(y_lshar_train, x_lshar_train).fit()

      
            oos_feat_date = X_1_log.index[oos_feature_index]
            oos_time = (oos_feat_date - global_start).total_seconds() / 604800
            oos_ls_features = pd.DataFrame(index=[oos_target_date])
            for n in periods:
                f = 1/n
                oos_ls_features[f'Sine_{n:.2f}'] = np.sin(2*np.pi*f*oos_time)
                oos_ls_features[f'Cos_{n:.2f}'] = np.cos(2*np.pi*f*oos_time)
            
            
            x_oos_har_no_const = oos_har_df.reset_index(drop=True)
            oos_ls_features_no_index = oos_ls_features.reset_index(drop=True)
            x_oos_lshar = pd.concat([x_oos_har_no_const, oos_ls_features_no_index], axis=1)
            x_oos_lshar = sm.add_constant(x_oos_lshar, has_constant='add') # Add constant back

            x_oos_lshar.columns = ols_results.params.index # Align columns to LSHAR params
            
            oos_log_pred = ols_results.predict(x_oos_lshar).iloc[0]
            sigma2_hybrid = ols_results.mse_resid
            forecast_rv_hybrid = np.exp(oos_log_pred + sigma2_hybrid / 2)
            

            oos_preds_1d_step_ahead[oos_target_date] = forecast_rv_hybrid
            oos_errors['QLIKE'][oos_target_date] = qlike_loss(actual_rv, forecast_rv_hybrid)
            oos_errors['RMSE'][oos_target_date] = (actual_rv - forecast_rv_hybrid)**2
            print(start)
            
        else:

            forecast_rv_hybrid = forecast_rv_har
            

        oos_errors_Hybrid['QLIKE'][oos_target_date] = qlike_loss(actual_rv, forecast_rv_hybrid)
        oos_errors_Hybrid['RMSE'][oos_target_date] = (actual_rv - forecast_rv_hybrid)**2
        
        start += step_size
        
    except Exception as e:
        print(f"Error for section-{start}: {e}")
        start += step_size
        continue


if oos_errors_HAR:
    final_qlike_har = np.mean(list(oos_errors_HAR['QLIKE'].values()))
    final_rmse_har = np.sqrt(np.mean(list(oos_errors_HAR['RMSE'].values())))
    total_har_preds = len(oos_errors_HAR['QLIKE'])

    final_qlike_hybrid = np.mean(list(oos_errors_Hybrid['QLIKE'].values()))
    final_rmse_hybrid = np.sqrt(np.mean(list(oos_errors_Hybrid['RMSE'].values())))
    
    
    print("Out-of-Sample Performance Comparison (Full Period)")
    print(f"Total Predictions (Full Period):** {total_har_preds}")
    print("---" * 15)
    
    print("HAR Baseline Model Performance")
    print(f"* Average QLIKE Loss: {final_qlike_har:.6f}")
    print(f"* Final RMSE: {final_rmse_har:.6f}")
    print("---" * 15)
    
    print("LSHAR/HAR Hybrid Model Performance")
    print(f"* Average QLIKE Loss: {final_qlike_hybrid:.6f}")
    print(f"* Final RMSE: {final_rmse_hybrid:.6f}")
    print("---" * 15)
    
    
    if oos_errors['QLIKE']:
        final_qlike_lshar_only = np.mean(list(oos_errors['QLIKE'].values()))
        final_rmse_lshar_only = np.sqrt(np.mean(list(oos_errors['RMSE'].values())))
        print("LSHAR-Only Model Performance (Diagnostic)")
        print(f"* **Predictions Count:** {len(oos_errors['QLIKE'])}")
        print(f"* Average QLIKE Loss: {final_qlike_lshar_only:.6f}")
        print(f"* Final RMSE: {final_rmse_lshar_only:.6f}")
        print("---" * 15)
else:
    print("No out-of-sample predictions were made.")

plt.plot(freq, power)
# plt.title("Lomb-Scargle Periodogram")
plt.xlabel("Frequency-->")
plt.ylabel("Power-->")
plt.axhline(fap_threshold, color='red')
plt.text(0.8, 0.0625, 'fap_threshold', bbox=dict(facecolor='red', alpha=0.5))

#%% LSHAR on HAR Residuals (Orthogonal Spectral Augmentation)
LS_periods = 0
lookback = 256
alpha = 0.1


results = []
try:
    curr_idx = X_1_log.index.get_indexer([start_date_train], method='bfill')[0]
except:
    curr_idx = X_1_log.index.searchsorted(start_date_train)

while curr_idx + lookback < len(X_1_log)-1:
    
    train_x = X_1_log.iloc[curr_idx : curr_idx + lookback]
    train_y = Y_1_log.iloc[curr_idx : curr_idx + lookback]
    
    oos_idx = curr_idx + lookback
    oos_x = X_1_log.iloc[[oos_idx]]
    
    
    oos_actual = RVd_full.iloc[oos_idx + 1]
    oos_date = X_1_log.index[oos_idx + 1]
    
    
    har_vars = ['rvd', 'rvw', 'rvm']
    model_har = sm.OLS(train_y, sm.add_constant(train_x[har_vars])).fit()
    sigma2_har = model_har.mse_resid # Use MSE for consistent variance
    
    log_pred_har = model_har.predict(sm.add_constant(oos_x[har_vars], has_constant='add')).iloc[0]
    pred_rv_har = np.exp(log_pred_har + sigma2_har/2)

    
    har_residuals = (train_y - model_har.fittedvalues).values.astype(np.float64)
    
    
    t_train = pd.to_timedelta(train_x.index - global_start).total_seconds().values / 86400.0
    
    ls = LombScargle(t_train, har_residuals)
    freq = np.linspace(1.0/lookback, 1/2, 1000) # Higher resolution for 512 lookback
    power = ls.power(freq)
    
    
    
    # fap_thresh = np.float64(mc_fap(t_train, har_residuals, freq, alpha))
    fap_thresh = np.float64(model.false_alarm_level(alpha))
    if hasattr(fap_thresh, "__len__"): fap_thresh = np.max(fap_thresh)
    
    periods = select_robust_periods(freq, power, fap_thresh, 0.005, 1.0)
    
    pred_rv_hybrid = pred_rv_har # Default
    
    if periods:
        LS_periods += 1
        ls_feats_train = get_ls_features(t_train, periods)
        model_ls = sm.OLS(har_residuals, sm.add_constant(ls_feats_train)).fit()
        
        
        t_oos = (oos_date - global_start).total_seconds() / 86400
        ls_feats_oos = get_ls_features(np.array([t_oos]), periods)
        
        log_resid_pred = model_ls.predict(sm.add_constant(ls_feats_oos, has_constant='add')).iloc[0]
        sigma2_hybrid = model_ls.mse_resid
        
        
        pred_rv_hybrid = np.exp(log_pred_har + log_resid_pred + sigma2_hybrid/2)

    results.append({
        'Date': oos_date,
        'Actual': oos_actual,
        'HAR_Pred': pred_rv_har,
        'Hybrid_Pred': pred_rv_hybrid
    })
    # print(curr_idx)
    curr_idx += step_size

plt.plot(freq, power)
# plt.title("Lomb-Scargle Periodogram")
plt.xlabel("Frequency-->")
plt.ylabel("Power-->")
plt.axhline(fap_thresh, color='red')
plt.text(0.4, 0.065, 'fap_threshold', bbox=dict(facecolor='red', alpha=0.5))
# plt.savefig(f"{file.replace(".csv","_residuals")}.png", dpi=300)

# --- 4. Final Evaluation ---
print(f"Number of LS_periods = {LS_periods}")
res_df = pd.DataFrame(results).set_index('Date')
res_df['HAR_QLIKE'] = qlike_loss(res_df['Actual'], res_df['HAR_Pred'])
res_df['Hybrid_QLIKE'] = qlike_loss(res_df['Actual'], res_df['Hybrid_Pred'])

res_df['HAR_SE'] = (res_df['Actual'] - res_df['HAR_Pred'])**2
res_df['Hybrid_SE'] = (res_df['Actual'] - res_df['Hybrid_Pred'])**2

print(f"HAR QLIKE: {res_df['HAR_QLIKE'].mean():.6f}")
print(f"Hybrid QLIKE: {res_df['Hybrid_QLIKE'].mean():.6f}")
print("-" * 30)
print(f"HAR RMSE: {np.sqrt(res_df['HAR_SE'].mean()):.6f}")
print(f"Hybrid RMSE: {np.sqrt(res_df['Hybrid_SE'].mean()):.6f}")