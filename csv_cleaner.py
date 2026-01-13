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
try:
    df['Datetime'] = pd.to_datetime(df['date'])
except:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
df.set_index('Datetime', inplace=True)

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