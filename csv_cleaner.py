#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:10:47 2026

@author: rbarcrosspbar
"""

import pandas as pd
import os
import tkinter as tk
from tkinter.filedialog import askdirectory

# ======================================================
# CONFIGURATION PARAMETERS
# ======================================================
MARKET_START = '09:15'
MARKET_END   = '15:25'
EXPECTED_BARS = 75        # 375 mins / 5 min bars = 75
REQUIRED_COLS = ['Open', 'High', 'Low', 'Close']


def has_valid_timestamps(group):
    """
    Accept a day-group only if it contains exactly the expected
    5-minute bars with no duplicates and no gaps.

    A day that has 75 rows but duplicate timestamps (and therefore
    a missing bar) will be rejected here, whereas the old len(x)==75
    check would have passed it silently.
    """
    date = group.index[0].date()
    expected = pd.date_range(
        start=f"{date} {MARKET_START}",
        end=f"{date} {MARKET_END}",
        freq="5min"
    )
    actual = group.index

    if len(actual) != EXPECTED_BARS:
        return False
    if actual.duplicated().any():
        return False
    if not actual.equals(expected):
        return False
    return True


def process_batch_stocks():
    # 1. Select Input and Output Folders
    root = tk.Tk()
    root.withdraw()

    input_folder = askdirectory(title="Select Folder Containing Raw CSVs")
    if not input_folder:
        print("No input folder selected.")
        return

    output_folder = askdirectory(title="Select/Create Folder for Cleaned CSVs")
    if not output_folder:
        print("No output folder selected.")
        return

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # 2. Loop through all CSV files
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print(f"Found {len(files)} CSV files. Starting processing...\n")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}...")

        try:
            # ── Load and Pre-process ──────────────────────────────────
            df = pd.read_csv(file_path)
            df.columns = [c.capitalize() for c in df.columns]

            if 'Date' in df.columns and 'Datetime' not in df.columns:
                df.rename(columns={'Date': 'Datetime'}, inplace=True)

            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
            df.sort_index(inplace=True)

            # ── Strip sub-minute noise so timestamps align to 5-min grid ──
            # e.g. 09:15:00.500 → 09:15:00  (safe no-op if already clean)
            df.index = df.index.floor("5min")

            # ── Intra-day Forward Fill (no cross-day leakage) ─────────
            df[REQUIRED_COLS] = df.groupby(df.index.date)[REQUIRED_COLS].ffill()

            # ── Filter Market Hours ───────────────────────────────────
            df_filtered = df.between_time(MARKET_START, MARKET_END).copy()

            df_final = (
                df_filtered
                .groupby(df_filtered.index.date)
                .filter(has_valid_timestamps)
                .copy()
            )

            # ── Save Output ───────────────────────────────────────────
            output_name = os.path.join(output_folder, f"Cleaned_{filename}")
            df_final.to_csv(output_name)

            days_before = len(df_filtered.index.normalize().unique())
            days_after  = len(df_final.index.normalize().unique())
            print(f"   - Done. Kept {days_after}/{days_before} days.")

        except Exception as e:
            print(f"   - Error processing {filename}: {e}")

    print("\n" + "=" * 30)
    print("BATCH PROCESSING COMPLETE")
    print(f"Cleaned files are in: {output_folder}")
    print("=" * 30)


if __name__ == "__main__":
    process_batch_stocks()
