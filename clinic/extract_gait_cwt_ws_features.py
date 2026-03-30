#!/usr/bin/env python3
"""
Extract clinic Gait(11), CWT(28), WalkSway(12) features from 6MWT.

Input:  csv_preprocessed2/*.csv, csv_raw2/*.csv
Output: feats/clinic_gait_features.csv (11 features x 101 subjects)
        feats/clinic_cwt_features.csv (28 features x 101 subjects)
        feats/clinic_walksway_features.csv (12 features x 101 subjects)

Run:  python temporary_experiments/extract_clinic_gait_cwt_ws_features.py
"""
import time, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = Path(__file__).parent.parent
PREPROC2 = BASE / 'csv_preprocessed2'
RAW = BASE / 'csv_raw2'

from clinic.reproduce_c2 import extract_gait10, compute_vt_rms, add_sway_ratios, extract_cwt
from clinic.extract_walking_sway import extract_walking_sway


def find_file(directory, cohort, subj_id):
    key = f'{cohort}{int(subj_id):02d}'
    for f in directory.glob(f'{key}_*.csv'):
        return f
    return None


if __name__ == '__main__':
    t0 = time.time()
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = (ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44]))
    ids101 = ids[~excl].reset_index(drop=True)
    n = len(ids101)
    print(f"Extracting clinic Gait/CWT/WalkSway for {n} subjects...")

    # Gait (11)
    gait_rows = []
    for _, r in ids101.iterrows():
        fp = find_file(PREPROC2, r['cohort'], r['subj_id'])
        gait_rows.append(extract_gait10(pd.read_csv(fp)))
    vt_rms_df = compute_vt_rms(PREPROC2)
    gdf = pd.DataFrame(gait_rows)
    gm = pd.concat([ids101.reset_index(drop=True), gdf], axis=1)
    sway = add_sway_ratios(gm.merge(vt_rms_df, on=['cohort', 'subj_id', 'sixmwd'], how='left'))
    gait_cols = ['cadence_hz', 'step_time_cv_pct', 'acf_step_regularity', 'hr_ap', 'hr_vt',
                 'ml_rms_g', 'ml_spectral_entropy', 'jerk_mean_abs_gps', 'enmo_mean_g',
                 'cadence_slope_per_min', 'vt_rms_g']

    # CWT (28)
    cwt_rows = []
    for _, r in ids101.iterrows():
        fp = find_file(RAW, r['cohort'], r['subj_id'])
        raw = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(np.float32)
        cwt_rows.append(extract_cwt(raw))

    # WalkSway (12)
    ws_rows = []
    for _, r in ids101.iterrows():
        fp = find_file(PREPROC2, r['cohort'], r['subj_id'])
        df = pd.read_csv(fp)
        ws_rows.append(extract_walking_sway(df['AP'].values, df['ML'].values, df['VT'].values))

    # Save
    FEATS = BASE / 'feats'
    keys = ids101.apply(lambda r: f"{r['cohort']}{int(r['subj_id']):02d}", axis=1).values

    gait_df = sway[gait_cols].copy()
    gait_df.insert(0, 'key', keys)
    gait_df.to_csv(FEATS / 'clinic_gait_features.csv', index=False)
    print(f"Saved feats/clinic_gait_features.csv ({gait_df.shape})")

    cwt_df = pd.DataFrame(cwt_rows)
    cwt_df.insert(0, 'key', keys)
    cwt_df.to_csv(FEATS / 'clinic_cwt_features.csv', index=False)
    print(f"Saved feats/clinic_cwt_features.csv ({cwt_df.shape})")

    ws10_df = pd.DataFrame(ws_rows)
    ws_ratios = sway[['ml_over_enmo', 'ml_over_vt']].reset_index(drop=True)
    ws_df = pd.concat([ws10_df, ws_ratios], axis=1)
    ws_df.insert(0, 'key', keys)
    ws_df.to_csv(FEATS / 'clinic_walksway_features.csv', index=False)
    print(f"Saved feats/clinic_walksway_features.csv ({ws_df.shape})")

    print(f"Done in {time.time()-t0:.0f}s")
