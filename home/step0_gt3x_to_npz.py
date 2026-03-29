#!/usr/bin/env python3
"""
Step 0: GT3X -> npz (one file per subject, full recording X,Y,Z + timestamps)
Saves to home_full_recording_npz/{key}.npz
No filtering applied — keeps entire recording as-is from device.
"""
import re, time
import numpy as np
import pandas as pd
from pathlib import Path
from pygt3x.reader import FileReader

BASE = Path(__file__).parent.parent
OUT_DIR = BASE / 'home_full_recording_npz'
OUT_DIR.mkdir(exist_ok=True)


def load_gt3x(path):
    with FileReader(str(path)) as reader:
        df = reader.to_pandas()
    ts = df.index.values  # timestamps from GT3X (epoch seconds, float64)
    xyz = df[['X', 'Y', 'Z']].values.astype(np.float32)
    return ts, xyz


def build_subject_list():
    ids = pd.read_csv(BASE / 'feats' / 'target_6mwd.csv')
    excl = (ids['cohort'] == 'M') & (ids['subj_id'].isin([22, 44]))
    ids101 = ids[~excl].reset_index(drop=True)

    accel = BASE / 'Accel files'
    gt3x_map = {}
    for d in accel.iterdir():
        if not d.is_dir(): continue
        m = re.match(r'^([CM])(\d+)', d.name)
        if not m: continue
        cohort, sid = m.group(1), int(m.group(2))
        key = f'{cohort}{sid:02d}'
        gt3x = list(d.glob('*.gt3x'))
        if gt3x:
            gt3x_map[key] = str(gt3x[0])

    subjects = []
    for _, r in ids101.iterrows():
        key = f"{r['cohort']}{int(r['subj_id']):02d}"
        if key not in gt3x_map:
            continue
        subjects.append({
            'key': key, 'cohort': r['cohort'], 'subj_id': int(r['subj_id']),
            'year': int(r['year']), 'sixmwd': int(r['sixmwd']),
            'gt3x_path': gt3x_map[key],
        })
    return pd.DataFrame(subjects)


if __name__ == '__main__':
    t0 = time.time()
    subj_df = build_subject_list()
    n = len(subj_df)
    print(f"Extracting full recording NPZ for {n} subjects...")

    for i, (_, r) in enumerate(subj_df.iterrows()):
        out_path = OUT_DIR / f"{r['key']}.npz"

        if out_path.exists():
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n}] {r['key']} (cached)")
            continue

        try:
            ts, xyz = load_gt3x(r['gt3x_path'])
            np.savez_compressed(out_path, xyz=xyz, timestamps=ts)
            mb = out_path.stat().st_size / 1e6
            print(f"  [{i+1}/{n}] {r['key']} -> {mb:.1f}MB ({len(xyz)} samples)")
        except Exception as ex:
            print(f"  [{i+1}/{n}] {r['key']} FAILED: {ex}")

    subj_df.to_csv(OUT_DIR / '_subjects.csv', index=False)
    print(f"\nSaved {n} files to {OUT_DIR}/")
    print(f"Done in {time.time()-t0:.0f}s")
