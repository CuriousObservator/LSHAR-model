#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 12:13:28 2025

@author: rbarcrosspbar
"""
import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() 
file = askopenfilename() 

# Read with header=0 to get the labels, but we will redefine them to match the 7 actual columns
# Columns in CSV: [Date, Time, Open, High, Low, Close, Volume]
df = pd.read_csv(file, header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])


df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df = df.drop(columns=['Date', 'Time'])


df = df.between_time('09:15', '15:25').copy()


counts = df.groupby(df.index.date).size()


if counts.empty:
    print("Error: No data found in the 09:15-15:25 window. Check your CSV time format.")
else:
    most_common_counts = counts.mode()
    if not most_common_counts.empty:
        most_common_count = most_common_counts[0]
        valid_dates = counts[counts >= (most_common_count - 5)].index
        df_filtered = df[np.isin(df.index.date, valid_dates)].copy()

        print(f"Removed {len(counts) - len(valid_dates)} anomaly days.")

        #Standardize to exactly 75 bars per day
        standard_times = pd.date_range("09:15", "15:25", freq="5min").time
        unique_dates = np.unique(df_filtered.index.date)

        full_index = [
            pd.Timestamp.combine(d, t) 
            for d in unique_dates 
            for t in standard_times
        ]

        df_final = df_filtered.reindex(full_index)


        df_final['Volume'] = df_final['Volume'].fillna(0)
        df_final['Close'] = df_final['Close'].ffill()
        for col in ['Open', 'High', 'Low']:
            df_final[col] = df_final[col].fillna(df_final['Close'])

        # Final Verification
        final_counts = df_final.groupby(df_final.index.date).size()
        print(f"Remaining days: {len(final_counts)}")
        print(f"Bars per day: {final_counts.unique()}")

        output_name = file.replace(".csv", "_Cleaned.csv")
        df_final.to_csv(output_name)
        print(f"Saved to: {output_name}")