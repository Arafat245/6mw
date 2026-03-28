#!/usr/bin/env python3
"""Add Timestamp column to all walking bout CSVs (30 Hz → 0, 1/30, 2/30, ...)."""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
BOUT_DIR = BASE / 'walking_bouts'
FS = 30

# Only process non-OPT subject directories (the ones used in our pipeline)
ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
excl = (ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44]))
ids101 = ids[~excl].reset_index(drop=True)

total = 0
for i, (_, r) in enumerate(ids101.iterrows()):
    subj_id = f"{r['cohort']}{int(r['subj_id']):02d}"
    subj_dir = BOUT_DIR / subj_id
    if not subj_dir.exists():
        print(f"  SKIP {subj_id}: no directory", flush=True)
        continue

    bout_files = sorted(subj_dir.glob('bout_*.csv'))
    for bf in bout_files:
        df = pd.read_csv(bf)
        if 'Timestamp' in df.columns:
            continue
        df.insert(0, 'Timestamp', np.arange(len(df)) / FS)
        df.to_csv(bf, index=False)
        total += 1

    print(f"  [{i+1}/101] {subj_id}: {len(bout_files)} bouts updated", flush=True)

print(f"\nDone. Added Timestamp to {total} bout files.")
