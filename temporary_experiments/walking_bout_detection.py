#!/usr/bin/env python3
"""
Detect walking bouts using AGD step counts, then extract raw acceleration
from GT3X files at those times. Save each bout with timestamps.

For each subject:
1. Read AGD file → find epochs with steps >= 5 → merge within 5s gap → walking bouts
2. Read GT3X file via actipy → get timestamped acceleration (30 Hz)
3. Slice GT3X data at walking bout times
4. Save each bout as CSV: Timestamp, X, Y, Z

Output: walking_bouts/{subject_folder}/bout_001_120s.csv, bout_002_45s.csv, ...
"""
import sys, warnings, time, re
import numpy as np
import pandas as pd
import sqlite3
import actipy
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent.parent
ACCEL_DIR = BASE / 'Accel files'
OUT_DIR = BASE / 'walking_bouts'

MIN_STEPS_PER_EPOCH = 5   # minimum steps in an epoch to count as walking
MERGE_GAP_SECONDS = 5     # merge bouts separated by ≤ 5 seconds
MIN_BOUT_SECONDS = 10     # minimum bout duration after merging


def ticks_to_datetime(ticks):
    """Convert .NET ticks to Python datetime."""
    return datetime(1, 1, 1) + timedelta(microseconds=ticks // 10)


def read_agd_walking_bouts(agd_path):
    """Read AGD file and detect walking bouts from step counts.

    1. Find all epochs with steps >= MIN_STEPS_PER_EPOCH
    2. Each walking epoch → (start, end) interval
    3. Merge intervals within MERGE_GAP_SECONDS
    4. Filter by MIN_BOUT_SECONDS

    Returns list of (start_datetime, end_datetime) tuples and epoch_sec."""
    conn = sqlite3.connect(str(agd_path))
    cursor = conn.cursor()

    cursor.execute("SELECT settingValue FROM settings WHERE settingName='epochlength'")
    epoch_sec = int(cursor.fetchone()[0])

    cursor.execute("SELECT dataTimestamp, steps FROM data ORDER BY dataTimestamp")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], epoch_sec

    # Step 1: Find all walking epoch intervals
    raw_intervals = []
    for ts_ticks, steps in rows:
        if steps >= MIN_STEPS_PER_EPOCH:
            dt = ticks_to_datetime(ts_ticks)
            raw_intervals.append((dt, dt + timedelta(seconds=epoch_sec)))

    if not raw_intervals:
        return [], epoch_sec

    # Step 2: Merge intervals within MERGE_GAP_SECONDS
    merged = [raw_intervals[0]]
    for start, end in raw_intervals[1:]:
        prev_start, prev_end = merged[-1]
        gap = (start - prev_end).total_seconds()
        if gap <= MERGE_GAP_SECONDS:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Step 3: Filter by minimum duration
    filtered = []
    for start, end in merged:
        dur = (end - start).total_seconds()
        if dur >= MIN_BOUT_SECONDS:
            filtered.append((start, end))

    return filtered, epoch_sec


def find_best_agd(accel_folder):
    """Find the best AGD file (prefer 60-sec epoch)."""
    agd_files = list(accel_folder.glob('*.agd'))
    if not agd_files:
        return None
    for f in agd_files:
        if '60sec' in f.name.lower():
            return f
    return agd_files[0]


def find_gt3x(accel_folder):
    """Find GT3X file in folder."""
    gt3x_files = list(accel_folder.glob('*.gt3x'))
    return gt3x_files[0] if gt3x_files else None


def load_gt3x(gt3x_path):
    """Load GT3X file using actipy. Returns DataFrame with datetime index and x,y,z columns."""
    data, info = actipy.read_device(
        str(gt3x_path),
        lowpass_hz=None,
        calibrate_gravity=False,
        detect_nonwear=False,
        resample_hz=None
    )
    return data


def extract_and_save_bouts(gt3x_data, bouts, out_folder):
    """Extract walking bout segments from GT3X data and save as CSV."""
    saved_count = 0

    for bout_idx, (bout_start, bout_end) in enumerate(bouts):
        # Convert to pandas Timestamps for slicing
        ts_start = pd.Timestamp(bout_start)
        ts_end = pd.Timestamp(bout_end)

        # Slice GT3X data at bout times
        bout_data = gt3x_data.loc[ts_start:ts_end]

        if len(bout_data) < MIN_BOUT_SECONDS * 30:  # need at least MIN_BOUT_SECONDS at ~30 Hz
            continue

        # Build output DataFrame with Unix timestamps
        timestamps = np.array([t.timestamp() for t in bout_data.index])
        bout_df = pd.DataFrame({
            'Timestamp': timestamps,
            'X': bout_data['x'].values,
            'Y': bout_data['y'].values,
            'Z': bout_data['z'].values,
        })

        dur_sec = len(bout_data) / 30
        out_fn = f"bout_{bout_idx+1:03d}_{dur_sec:.0f}s.csv"
        bout_df.to_csv(out_folder / out_fn, index=False)
        saved_count += 1

    return saved_count


if __name__ == '__main__':
    t0 = time.time()
    print("Walking Bout Detection: AGD step counts → GT3X raw acceleration")
    print("=" * 70)

    folders = sorted([d for d in ACCEL_DIR.iterdir() if d.is_dir()])
    total_bouts = 0
    total_subjects = 0
    skipped = []
    summary = []

    for folder in folders:
        subject_name = folder.name

        # Find AGD
        agd_path = find_best_agd(folder)
        if agd_path is None:
            skipped.append((subject_name, 'no AGD'))
            continue

        # Find GT3X
        gt3x_path = find_gt3x(folder)
        if gt3x_path is None:
            skipped.append((subject_name, 'no GT3X'))
            continue

        # Detect walking bouts from AGD
        bouts, epoch_sec = read_agd_walking_bouts(agd_path)
        if not bouts:
            skipped.append((subject_name, 'no walking bouts'))
            continue

        # Load GT3X
        try:
            gt3x_data = load_gt3x(gt3x_path)
        except Exception as e:
            skipped.append((subject_name, f'GT3X read error: {e}'))
            continue

        # Output folder
        out_folder = OUT_DIR / subject_name
        out_folder.mkdir(parents=True, exist_ok=True)

        # Extract and save
        n_saved = extract_and_save_bouts(gt3x_data, bouts, out_folder)
        total_bouts += n_saved
        total_subjects += 1

        # Compute stats
        bout_durs = [(b[1] - b[0]).total_seconds() for b in bouts]
        summary.append({
            'subject': subject_name,
            'agd_bouts': len(bouts),
            'saved_bouts': n_saved,
            'total_walk_min': sum(bout_durs) / 60,
            'longest_bout_sec': max(bout_durs),
            'epoch_sec': epoch_sec,
        })

        print(f"  [{total_subjects:3d}] {subject_name:20s}  AGD:{len(bouts):>4d} bouts → "
              f"saved:{n_saved:>4d}  total:{sum(bout_durs)/60:.0f}min  "
              f"longest:{max(bout_durs):.0f}s  (epoch={epoch_sec}s)", flush=True)

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / 'walking_bout_summary.csv', index=False)

    print(f"\n{'='*70}")
    print(f"Done in {time.time()-t0:.0f}s")
    print(f"Processed: {total_subjects} subjects, {total_bouts} total bouts saved")
    print(f"Summary saved to walking_bouts/walking_bout_summary.csv")

    if skipped:
        print(f"\nSkipped: {len(skipped)} subjects")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
