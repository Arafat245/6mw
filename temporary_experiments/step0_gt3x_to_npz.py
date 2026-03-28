#!/usr/bin/env python3
"""
Step 0: GT3X -> npz (one file per subject, daytime X,Y,Z as float32)
Saves to csv_home_daytime_npz/{key}.npz  (~5-10MB each vs ~400MB CSV)
Run with pygt3x-env.
"""
import os, re, pickle, time
import numpy as np
import pandas as pd
from pathlib import Path
from pygt3x.reader import FileReader

BASE = Path(__file__).parent.parent
OUT_DIR = BASE / 'csv_home_daytime_npz'
OUT_DIR.mkdir(exist_ok=True)
FS = 30


def load_gt3x(path):
    with FileReader(str(path)) as reader:
        df = reader.to_pandas()
    ts = df.index.values.astype(np.int64) // 10**9
    xyz = df[['X', 'Y', 'Z']].values.astype(np.float32)
    return ts, xyz


def extract_daytime(timestamps, xyz, day_start=7, day_end=22):
    hours = (timestamps % 86400) / 3600
    daytime = (hours >= day_start) & (hours < day_end)
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    win = int(5 * FS)
    if len(vm) > win:
        rstd = pd.Series(vm).rolling(win, center=True).std().values
        rstd = np.nan_to_num(rstd, nan=0)
        worn = rstd > 0.01
    else:
        worn = np.ones(len(vm), dtype=bool)
    return xyz[daytime & worn]


def build_subject_list():
    with open(BASE / '6mw_segmented_walk_data_dict.pkl', 'rb') as f:
        pkl = pickle.load(f)

    accel = BASE / 'Accel files'
    gt3x_map = {}
    for d in accel.iterdir():
        if not d.is_dir(): continue
        m = re.match(r'^([CM])(\d+)', d.name)
        if not m: continue
        cohort, sid = m.group(1), int(m.group(2))
        key = f'{cohort}{sid:02d}' if sid < 100 else f'{cohort}{sid}'
        gt3x = list(d.glob('*.gt3x'))
        if gt3x:
            gt3x_map[key] = str(gt3x[0])

    excl = {'M22', 'M44'}
    items = [(k, v) for k, v in pkl.items() if k not in excl and k in gt3x_map]
    c = sorted([x for x in items if x[0].startswith('C')], key=lambda x: int(x[0][1:]))
    m_list = sorted([x for x in items if x[0].startswith('M')], key=lambda x: int(x[0][1:]))
    subjects = []
    for key, val in c + m_list:
        cohort = key[0]
        sid = int(key[1:])
        ym = re.search(r'\((\d{4})', gt3x_map[key])
        year = int(ym.group(1)) if ym else 2016
        subjects.append({
            'key': key, 'cohort': cohort, 'subj_id': sid,
            'year': year, 'sixmwd': int(val['distance']),
            'gt3x_path': gt3x_map[key],
        })
    return pd.DataFrame(subjects)


if __name__ == '__main__':
    t0 = time.time()
    subj_df = build_subject_list()
    n = len(subj_df)
    print(f"Extracting daytime NPZ for {n} subjects...")

    for i, (_, r) in enumerate(subj_df.iterrows()):
        out_path = OUT_DIR / f"{r['key']}.npz"

        if out_path.exists():
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n}] {r['key']} (cached)")
            continue

        try:
            ts, xyz = load_gt3x(r['gt3x_path'])
            daytime = extract_daytime(ts, xyz)
            if len(daytime) < int(60 * FS):
                daytime = xyz
            np.savez_compressed(out_path, xyz=daytime)
            hrs = len(daytime) / FS / 3600
            mb = out_path.stat().st_size / 1e6
            print(f"  [{i+1}/{n}] {r['key']} -> {mb:.1f}MB ({hrs:.1f}h, {len(daytime)} samples)")
        except Exception as ex:
            print(f"  [{i+1}/{n}] {r['key']} FAILED: {ex}")

    # Save subject list as well
    subj_df.to_csv(OUT_DIR / '_subjects.csv', index=False)
    print(f"\nSaved {n} files to {OUT_DIR}/")
    print(f"Done in {time.time()-t0:.0f}s")
