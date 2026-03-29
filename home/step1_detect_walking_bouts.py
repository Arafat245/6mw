#!/usr/bin/env python3
"""
Step 1: Detect walking bouts from full recording NPZ files.
Optionally saves each bout as CSV with Timestamp, X, Y, Z.

Input:  home_full_recording_npz/*.npz
Output: feats/home_walking_bouts.pkl (bout indices per subject)
        walking_bouts/{subject_id}/bout_*.csv (optional, with --save-csv)

Run:  python home/step1_detect_walking_bouts.py [--save-csv]
"""
import argparse, time
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.signal import butter, filtfilt

BASE = Path(__file__).parent.parent
NPZ_DIR = BASE / 'home_full_recording_npz'
FS = 30


def detect_walking_bouts(xyz, fs, min_bout_sec=10, merge_gap_sec=5):
    """
    Three-stage walking bout detection:
      Stage 1: ENMO >= 0.015g per second, min 10s bouts
      Stage 2: Harmonic ratio >= 0.2 in 10s windows
      Stage 3: Merge adjacent bouts within 5s gap
    """
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    enmo = np.maximum(vm - 1.0, 0.0)
    sec = int(fs); n_secs = len(enmo) // sec
    if n_secs < min_bout_sec:
        return []
    enmo_sec = enmo[:n_secs * sec].reshape(n_secs, sec).mean(axis=1)
    active = enmo_sec >= 0.015

    # Stage 1: active bouts
    raw_bouts = []
    in_b, bs = False, 0
    for s in range(n_secs):
        if active[s] and not in_b: bs = s; in_b = True
        elif not active[s] and in_b:
            if s - bs >= min_bout_sec: raw_bouts.append((bs * sec, s * sec))
            in_b = False
    if in_b and n_secs - bs >= min_bout_sec:
        raw_bouts.append((bs * sec, n_secs * sec))
    if not raw_bouts:
        return []

    # Stage 2: HR refinement
    b_filt, a_filt = butter(4, [0.5, 3.0], btype='bandpass', fs=fs)
    vm_bp = filtfilt(b_filt, a_filt, vm - vm.mean())
    win = int(10 * fs); step = int(10 * fs)
    fft_freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band = (fft_freqs >= 0.8) & (fft_freqs <= 3.5)
    refined = []
    for bout_s, bout_e in raw_bouts:
        walking_wins = []
        for wi in range(bout_s, bout_e - win, step):
            seg = vm_bp[wi:wi + win]
            X = np.fft.rfft(seg); mags = np.abs(X)
            if not np.any(band): continue
            cadence = fft_freqs[band][np.argmax(mags[band])]
            even, odd = 0.0, 0.0
            for k in range(1, 11):
                fk = k * cadence
                if fk >= fft_freqs[-1]: break
                idx = int(np.argmin(np.abs(fft_freqs - fk)))
                if k % 2 == 0: even += mags[idx]
                else: odd += mags[idx]
            hr = even / (odd + 1e-12) if odd > 0 else 0
            if hr >= 0.2: walking_wins.append((wi, wi + win))
        if walking_wins:
            cs, ce = walking_wins[0]
            for ws, we in walking_wins[1:]:
                if ws <= ce + step: ce = max(ce, we)
                else:
                    if ce - cs >= min_bout_sec * fs: refined.append((cs, ce))
                    cs, ce = ws, we
            if ce - cs >= min_bout_sec * fs: refined.append((cs, ce))
        else:
            if (bout_e - bout_s) >= min_bout_sec * fs:
                refined.append((bout_s, bout_e))
    if not refined:
        return []

    # Stage 3: merge adjacent
    merged = [refined[0]]
    for s, e in refined[1:]:
        prev_s, prev_e = merged[-1]
        if (s - prev_e) / fs <= merge_gap_sec:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-csv', action='store_true', help='Save each bout as CSV')
    args = parser.parse_args()

    t0 = time.time()
    subj_df = pd.read_csv(NPZ_DIR / '_subjects.csv')
    n = len(subj_df)
    print(f"Detecting walking bouts for {n} subjects...")

    BOUT_DIR = BASE / 'walking_bouts'
    all_bouts = {}
    total_bouts = 0

    for i, (_, r) in enumerate(subj_df.iterrows()):
        npz_path = NPZ_DIR / f"{r['key']}.npz"
        if not npz_path.exists():
            print(f"  WARNING: {r['key']} NPZ missing")
            continue

        data = np.load(npz_path)
        xyz = data['xyz'].astype(np.float64)
        bouts = detect_walking_bouts(xyz, FS, min_bout_sec=10, merge_gap_sec=5)
        all_bouts[r['key']] = bouts

        if args.save_csv and bouts:
            ts = data['timestamps'] if 'timestamps' in data else np.arange(len(xyz)) / FS
            subj_dir = BOUT_DIR / r['key']
            subj_dir.mkdir(parents=True, exist_ok=True)
            for bout_idx, (s, e) in enumerate(bouts):
                dur_sec = (e - s) / FS
                pd.DataFrame({
                    'Timestamp': ts[s:e],
                    'X': xyz[s:e, 0], 'Y': xyz[s:e, 1], 'Z': xyz[s:e, 2],
                }).to_csv(subj_dir / f'bout_{bout_idx+1:04d}_{dur_sec:.0f}s.csv', index=False)

        total_bouts += len(bouts)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] {r['key']}: {len(bouts)} bouts", flush=True)

    # Save bout indices
    FEATS_DIR = BASE / 'feats'
    with open(FEATS_DIR / 'home_walking_bouts.pkl', 'wb') as f:
        pickle.dump({'bouts': all_bouts}, f)
    print(f"\nSaved feats/home_walking_bouts.pkl ({len(all_bouts)} subjects, {total_bouts} total bouts)")
    if args.save_csv:
        print(f"Saved {total_bouts} bout CSVs to walking_bouts/")
    print(f"Done in {time.time()-t0:.0f}s")
