#!/usr/bin/env python3
"""
Step 1: GT3X → Daytime Extraction → Walking Bout Detection → Save
===================================================================
Processes raw GT3X files from Accel files/ and saves:
  - daytime_segments/  (raw X,Y,Z daytime data per subject)
  - walking_segments/  (preprocessed AP,ML,VT walking bouts per subject)

Run this ONCE. Then use run_all_models.py for prediction experiments.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.decomposition import PCA
from pygt3x.reader import FileReader

BASE = Path(__file__).parent
OUT = BASE / "results_raw_pipeline"
DAYTIME_DIR = OUT / "daytime_segments"
WALK_DIR = OUT / "walking_segments"
for d in [OUT, DAYTIME_DIR, WALK_DIR]:
    d.mkdir(exist_ok=True)

FS = 30.0


def load_gt3x(path):
    with FileReader(str(path)) as reader:
        df = reader.to_pandas()
    return df.index.values, df[["X", "Y", "Z"]].values.astype(np.float32)


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


def detect_walking_bouts(xyz, fs=FS, win_sec=10, step_sec=2, min_bout_sec=20):
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    is_walk, starts = [], []
    for s in range(0, len(vm) - win, step):
        seg_vm = vm[s:s + win]
        rms = np.sqrt(np.mean(seg_vm**2))
        std = np.std(seg_vm)
        seg = xyz[s:s + win]
        var = [np.var(seg[:, i]) for i in range(3)]
        best = seg[:, np.argmax(var)]
        bc = best - best.mean()
        acf = np.correlate(bc, bc, "full")[len(bc) - 1:]
        acf /= (acf[0] + 1e-12)
        search = acf[int(0.3 * fs):int(1.5 * fs)]
        peaks, props = find_peaks(search, height=0.1)
        reg = props["peak_heights"][0] if len(peaks) > 0 else 0
        is_walk.append((std > 0.05) and (rms > 0.8) and (rms < 1.5) and (reg > 0.15))
        starts.append(s)

    bouts = []
    in_b, bs = False, 0
    for i, (w, s) in enumerate(zip(is_walk, starts)):
        if w and not in_b:
            bs = s; in_b = True
        elif not w and in_b:
            be = starts[i - 1] + win
            if (be - bs) >= min_bout_sec * fs:
                bouts.append((bs, be))
            in_b = False
    if in_b:
        be = starts[-1] + win
        if (be - bs) >= min_bout_sec * fs:
            bouts.append((bs, be))
    return bouts


def preprocess_bout(seg, fs=FS):
    if len(seg) < 50:
        return seg
    b, a = butter(4, 0.25 / (fs / 2), btype="low")
    g_est = filtfilt(b, a, seg, axis=0)
    g_mean = np.mean(g_est, axis=0)
    g_dir = g_mean / (np.linalg.norm(g_mean) + 1e-12)
    g_proj = (seg @ g_dir)[:, None] * g_dir[None, :]
    dyn = seg - g_proj
    vt = seg @ g_dir - np.mean(seg @ g_dir)
    acc_h = dyn - (dyn @ g_dir)[:, None] * g_dir[None, :]
    if acc_h.shape[0] > 10:
        pca = PCA(n_components=2)
        h2d = pca.fit_transform(acc_h)
        ap, ml = h2d[:, 0], h2d[:, 1]
    else:
        ap, ml = acc_h[:, 0], acc_h[:, 1]
    return np.column_stack([ap, ml, vt])


def build_subject_list():
    home = pd.read_csv(BASE / "sway_features_home.csv")
    home.rename(columns={"year_x": "year"}, inplace=True)
    accel_dir = BASE / "Accel files"
    fmap = {}
    for folder in accel_dir.iterdir():
        if not folder.is_dir():
            continue
        m = re.match(r"([CM])(\d+)", folder.name)
        if m:
            fmap[(m.group(1), int(m.group(2)))] = folder

    subjects = []
    for _, row in home.iterrows():
        key = (row["cohort"], int(row["subj_id"]))
        folder = fmap.get(key)
        gt3x = list(folder.glob("*.gt3x")) if folder else []
        subjects.append({
            "cohort": row["cohort"], "subj_id": int(row["subj_id"]),
            "year": int(row["year"]), "sixmwd": int(row["sixmwd"]),
            "gt3x_path": str(gt3x[0]) if gt3x else None,
        })
    return pd.DataFrame(subjects)


def main():
    print("=" * 60)
    print("Preprocessing: GT3X → Daytime → Walking → Save")
    print("=" * 60)

    subjects = build_subject_list()
    n = len(subjects)
    print(f"  {n} subjects")

    for i, (_, row) in enumerate(subjects.iterrows()):
        c, s, yr, d = row["cohort"], row["subj_id"], row["year"], row["sixmwd"]
        fname = f"{c}{s:02d}_{yr}_{d}.csv"

        # Skip if already processed
        walk_path = WALK_DIR / fname
        daytime_path = DAYTIME_DIR / fname
        if walk_path.exists() and daytime_path.exists():
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{n} (cached)", flush=True)
            continue

        # Load GT3X
        timestamps, xyz = load_gt3x(row["gt3x_path"])

        # Daytime extraction
        daytime = extract_daytime(timestamps, xyz)
        if len(daytime) < int(60 * FS):
            daytime = xyz
        pd.DataFrame(daytime, columns=["X", "Y", "Z"]).to_csv(daytime_path, index=False)

        # Walking detection
        bouts = detect_walking_bouts(daytime)
        if bouts:
            segs = [preprocess_bout(daytime[s:e]) for s, e in bouts[:20]]
            walk = np.concatenate(segs, axis=0)
        else:
            walk = preprocess_bout(daytime[:int(600 * FS)])
        pd.DataFrame(walk, columns=["AP", "ML", "VT"]).to_csv(walk_path, index=False)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n} (bouts={len(bouts)})", flush=True)

    print(f"    {n}/{n}")
    print(f"  Daytime: {DAYTIME_DIR}/")
    print(f"  Walking: {WALK_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
