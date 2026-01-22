#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 12:13:28 2025

@author: rbarcrosspbar
"""
import pandas as pd
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() 
file = askopenfilename() 

df = pd.read_csv(file)
df = pd.read_csv(file)
if 'Datetime' not in df.columns and 'date' not in df.columns:
    df = pd.read_csv(file, header=None, names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Standardize column name
if 'date' in df.columns:
    df.rename(columns={'date': 'Datetime'}, inplace=True)
    
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')

df = df.between_time('09:15', '15:25').copy()

BARS_PER_DAY = 75
df_final = df.groupby(df.index.date).filter(lambda x: len(x) == BARS_PER_DAY).copy()

final_counts = df_final.groupby(df_final.index.date).size()
total_days_pre_filter = len(df.groupby(df.index.date))
removed_days = total_days_pre_filter - len(final_counts)

print(f"Total days analyzed: {total_days_pre_filter}")
print(f"Removed {removed_days} incomplete days.")
print(f"Remaining consistent days: {len(final_counts)}")


output_name = f"{os.path.splitext(file)[0]}_Strict_Cleaned.csv"
df_final.to_csv(output_name)
print(f"Saved to: {output_name}")