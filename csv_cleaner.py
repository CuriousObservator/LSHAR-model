#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 12:13:28 2025

@author: rbarcrosspbar
"""
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() 
file = askopenfilename() 

# 1. Load and initial cleanup
df = pd.read_csv(file, header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df = df.drop(columns=['Date', 'Time'])

# 2. Filter for standard market hours (09:15 to 15:25)
df = df.between_time('09:15', '15:25').copy()

# 3. Drop days with missing bars to maintain consistency
# We define a 'perfect day' as having exactly 75 five-minute bars
# (9:15 to 15:25 inclusive is 75 bars)
BARS_PER_DAY = 75
df_final = df.groupby(df.index.date).filter(lambda x: len(x) == BARS_PER_DAY).copy()

# Final Verification
final_counts = df_final.groupby(df_final.index.date).size()
removed_days = len(df.groupby(df.index.date)) - len(final_counts)

print(f"Removed {removed_days} incomplete days.")
print(f"Remaining consistent days: {len(final_counts)}")
print(f"Bars per day (verified): {final_counts.unique()}")

# Save the strictly cleaned data
output_name = file.replace(".csv", "_Strict_Cleaned.csv")
df_final.to_csv(output_name)
print(f"Saved to: {output_name}")