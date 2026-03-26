#!/usr/bin/env python3
"""
Compare walking detection methods for A1 performance:
  1. Current heuristic (RMS × autocorrelation regularity, top 25%)
  2. Validated algorithm based on Karas et al. 2023 (npj Digital Medicine)
     "one-size-fits-most" walking recognition:
     - Amplitude threshold on vector magnitude (peak-to-peak > threshold)
     - Dominant frequency in walking range (1.4-2.3 Hz)
     - Minimum bout duration ≥ 3 consecutive seconds
     Validated: sensitivity 0.95, specificity >0.95 at waist placement.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.fft import rfft, rfftfreq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
FS = 30.0


# ── Validated walking detection (Karas et al. 2023 adapted) ──────────────

def validated_walking_detection(sig, fs=FS, amp_thresh=0.15,
                                 freq_lo=1.4, freq_hi=2.3,
                                 min_bout_sec=3):
    """Validated walking bout detection adapted from Karas et al. 2023.

    For preprocessed AP/ML/VT signals (gravity removed), uses:
    1. Amplitude: peak-to-peak VM in 1-sec windows > amp_thresh
    2. Frequency: dominant frequency of VT in walking range (1.4-2.3 Hz)
    3. Duration: ≥ min_bout_sec consecutive walking seconds

    Returns list of (start_sample, end_sample) for detected walking bouts.
    """
    vm = np.sqrt(sig[:, 0]**2 + sig[:, 1]**2 + sig[:, 2]**2)
    vt = sig[:, 2]
    win = int(fs)  # 1-second windows
    n_windows = len(vm) // win

    is_walking = np.zeros(n_windows, dtype=bool)

    for i in range(n_windows):
        s = i * win
        e = s + win

        # 1. Amplitude check: peak-to-peak VM > threshold
        seg_vm = vm[s:e]
        p2p = seg_vm.max() - seg_vm.min()
        if p2p < amp_thresh:
            continue

        # 2. Frequency check: dominant freq of VT in walking range
        seg_vt = vt[s:e]
        if len(seg_vt) < win:
            continue

        # Use FFT to find dominant frequency
        seg_c = seg_vt - seg_vt.mean()
        fft_vals = np.abs(rfft(seg_c))
        fft_freqs = rfftfreq(len(seg_c), 1.0 / fs)

        # Find peak in walking frequency range
        walk_band = (fft_freqs >= freq_lo) & (fft_freqs <= freq_hi)
        if not walk_band.any():
            continue

        walk_power = fft_vals[walk_band].max()
        total_power = fft_vals[1:].max() + 1e-12  # exclude DC

        # Walking if dominant energy is in walking band
        if walk_power / total_power > 0.3:
            is_walking[i] = True

    # 3. Duration: merge consecutive walking windows into bouts (≥ min_bout_sec)
    bouts = []
    in_bout = False
    bout_start = 0
    for i in range(n_windows):
        if is_walking[i] and not in_bout:
            bout_start = i
            in_bout = True
        elif not is_walking[i] and in_bout:
            if (i - bout_start) >= min_bout_sec:
                bouts.append((bout_start * win, i * win))
            in_bout = False
    if in_bout and (n_windows - bout_start) >= min_bout_sec:
        bouts.append((bout_start * win, n_windows * win))

    return bouts


def validated_walking_detection_raw(sig_xyz, fs=FS, amp_thresh=0.3,
                                      freq_lo=1.4, freq_hi=2.3,
                                      min_bout_sec=3):
    """Validated walking detection on raw X,Y,Z (Karas et al. 2023).
    Uses 0.3g amplitude threshold as in the original paper.
    """
    vm = np.sqrt(sig_xyz[:, 0]**2 + sig_xyz[:, 1]**2 + sig_xyz[:, 2]**2)
    win = int(fs)
    n_windows = len(vm) // win

    is_walking = np.zeros(n_windows, dtype=bool)

    for i in range(n_windows):
        s = i * win
        e = s + win
        seg_vm = vm[s:e]

        # 1. Amplitude: peak-to-peak > 0.3g
        p2p = seg_vm.max() - seg_vm.min()
        if p2p < amp_thresh:
            continue

        # 2. Frequency: dominant freq in walking range
        # Use the axis with most variance (typically vertical)
        seg = sig_xyz[s:e]
        variances = [np.var(seg[:, j]) for j in range(3)]
        best_axis = seg[:, np.argmax(variances)]
        seg_c = best_axis - best_axis.mean()
        fft_vals = np.abs(rfft(seg_c))
        fft_freqs = rfftfreq(len(seg_c), 1.0 / fs)

        walk_band = (fft_freqs >= freq_lo) & (fft_freqs <= freq_hi)
        if not walk_band.any():
            continue

        walk_power = fft_vals[walk_band].max()
        total_power = fft_vals[1:].max() + 1e-12
        if walk_power / total_power > 0.3:
            is_walking[i] = True

    bouts = []
    in_bout = False
    bout_start = 0
    for i in range(n_windows):
        if is_walking[i] and not in_bout:
            bout_start = i
            in_bout = True
        elif not is_walking[i] and in_bout:
            if (i - bout_start) >= min_bout_sec:
                bouts.append((bout_start * win, i * win))
            in_bout = False
    if in_bout and (n_windows - bout_start) >= min_bout_sec:
        bouts.append((bout_start * win, n_windows * win))

    return bouts


# ── Current heuristic (for comparison) ──────────────────────────────────

def heuristic_walking_selection(sig, fs=FS, top_pct=0.25, win_sec=10, step_sec=2):
    """Current heuristic: score windows by RMS × (1 + autocorr regularity), take top 25%."""
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    vt = sig[:, 2]
    vm = np.sqrt(sig[:, 0]**2 + sig[:, 1]**2 + sig[:, 2]**2)
    scores, starts = [], []
    for s in range(0, len(vt) - win, step):
        rms = np.sqrt(np.mean(vm[s:s + win]**2))
        seg_c = vt[s:s + win] - vt[s:s + win].mean()
        acf = np.correlate(seg_c, seg_c, "full")[len(seg_c) - 1:]
        acf /= (acf[0] + 1e-12)
        search = acf[int(0.3 * fs):int(1.5 * fs)]
        peaks, props = find_peaks(search, height=0.05)
        reg = props["peak_heights"][0] if len(peaks) > 0 else 0
        scores.append(rms * (1 + reg))
        starts.append(s)
    scores = np.array(scores)
    n_sel = max(5, int(len(scores) * top_pct))
    n_sel = min(n_sel, len(scores))
    top_idx = np.sort(np.argsort(scores)[-n_sel:])
    return np.concatenate([sig[starts[i]:starts[i] + win] for i in top_idx])


# ── Preprocessing for raw signals ───────────────────────────────────────

def preprocess_segment(seg_xyz, fs=FS):
    if len(seg_xyz) < 50:
        return seg_xyz
    b, a = butter(4, 0.25 / (fs / 2), btype="low")
    g_est = filtfilt(b, a, seg_xyz, axis=0)
    g_mean = np.mean(g_est, axis=0)
    g_dir = g_mean / (np.linalg.norm(g_mean) + 1e-12)
    g_proj = (seg_xyz @ g_dir)[:, None] * g_dir[None, :]
    dyn = seg_xyz - g_proj
    vt = seg_xyz @ g_dir - np.mean(seg_xyz @ g_dir)
    acc_h = dyn - (dyn @ g_dir)[:, None] * g_dir[None, :]
    from sklearn.decomposition import PCA
    if acc_h.shape[0] > 10:
        pca = PCA(n_components=2)
        h2d = pca.fit_transform(acc_h)
        ap, ml = h2d[:, 0], h2d[:, 1]
    else:
        ap, ml = acc_h[:, 0], acc_h[:, 1]
    return np.column_stack([ap, ml, vt])


# ── Gait feature extraction (same 35 features) ─────────────────────────

def extract_gait_features(sig, fs=FS):
    ap, ml, vt = sig[:, 0], sig[:, 1], sig[:, 2]
    vm = np.sqrt(ap**2 + ml**2 + vt**2)
    f = {}
    for nm, ax in [("ap", ap), ("ml", ml), ("vt", vt), ("vm", vm)]:
        f[f"{nm}_rms"] = np.sqrt(np.mean(ax**2))
        f[f"{nm}_std"] = np.std(ax)
        f[f"{nm}_iqr"] = np.percentile(ax, 75) - np.percentile(ax, 25)
    f["sma"] = np.mean(np.abs(ap) + np.abs(ml) + np.abs(vt))
    for nm, ax in [("ap", ap), ("ml", ml), ("vt", vt)]:
        f[f"{nm}_jerk_rms"] = np.sqrt(np.mean((np.diff(ax) * fs)**2))
    for nm, ax in [("vt", vt), ("ap", ap)]:
        ax_c = ax - ax.mean()
        acf = np.correlate(ax_c, ax_c, "full")[len(ax_c) - 1:]
        acf /= (acf[0] + 1e-12)
        ml2 = min(int(1.5 * fs), len(acf) - 1)
        search = acf[int(0.3 * fs):ml2]
        peaks, props = find_peaks(search, height=0.0)
        f[f"{nm}_step_reg"] = props["peak_heights"][0] if len(peaks) >= 1 else 0
        f[f"{nm}_step_time"] = (peaks[0] + int(0.3 * fs)) / fs if len(peaks) >= 1 else 0
        f[f"{nm}_stride_reg"] = props["peak_heights"][1] if len(peaks) >= 2 else 0
        f[f"{nm}_step_sym"] = (f[f"{nm}_step_reg"] / (f[f"{nm}_stride_reg"] + 1e-8)
                               if len(peaks) >= 2 else 1.0)
    for nm, ax in [("vt", vt), ("ap", ap), ("ml", ml)]:
        if len(ax) > 64:
            freqs, psd = welch(ax, fs=fs, nperseg=min(256, len(ax)))
            gb = (freqs >= 0.5) & (freqs <= 3.5)
            f[f"{nm}_dom_freq"] = freqs[gb][np.argmax(psd[gb])] if gb.any() else 0
            gp = np.trapz(psd[gb], freqs[gb]) if gb.any() else 0
            tp = np.trapz(psd, freqs) + 1e-12
            f[f"{nm}_gait_pwr"] = gp / tp
            pn = psd / (psd.sum() + 1e-12); pn = pn[pn > 0]
            f[f"{nm}_spec_ent"] = -np.sum(pn * np.log2(pn + 1e-12))
        else:
            f[f"{nm}_dom_freq"] = 0; f[f"{nm}_gait_pwr"] = 0; f[f"{nm}_spec_ent"] = 0
    fund = f.get("vt_dom_freq", 0)
    if fund > 0:
        fv = np.abs(rfft(vt)); ff = rfftfreq(len(vt), 1 / fs)
        ep, op = 0, 0
        for h in range(1, 11):
            idx = np.argmin(np.abs(ff - h * fund))
            if h % 2 == 0: ep += fv[idx]**2
            else: op += fv[idx]**2
        f["vt_hr"] = ep / (op + 1e-12)
    else:
        f["vt_hr"] = 0
    win2 = int(2 * fs)
    rms_w = np.array([np.sqrt(np.mean(vm[i:i + win2]**2))
                      for i in range(0, len(vm) - win2 + 1, win2)])
    f["act_rms_cv"] = rms_w.std() / (rms_w.mean() + 1e-12)
    return f


# ── Main comparison ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Comparing walking detection methods (A1 config, LOO CV)")
    print("  1. Current heuristic (top 25% by RMS×regularity)")
    print("  2. Validated (Karas et al. 2023) on preprocessed AP/ML/VT")
    print("  3. Validated (Karas et al. 2023) on raw X/Y/Z from csv_ca")
    print("=" * 70)

    home = pd.read_csv(os.path.join(BASE, "sway_features_home.csv"))
    home.rename(columns={"year_x": "year"}, inplace=True)
    y = home["sixmwd"].values.astype(float)
    n = len(y)

    # Method 1: Current heuristic on preprocessed
    print("\n[1/3] Current heuristic (preprocessed csv_processed_home)...")
    rows1 = []
    for i, (_, row) in enumerate(home.iterrows()):
        c, sid, yr, d = row["cohort"], int(row["subj_id"]), int(row["year"]), int(row["sixmwd"])
        fname = f"{c}{sid:02d}_{yr}_{d}.csv"
        df = pd.read_csv(os.path.join(BASE, "csv_processed_home", fname))
        sig = df[["AP", "ML", "VT"]].values.astype(np.float32)
        walk = heuristic_walking_selection(sig)
        rows1.append(extract_gait_features(walk))
        if (i + 1) % 50 == 0: print(f"  {i+1}/{n}", flush=True)
    print(f"  {n}/{n}")

    # Method 2: Validated on preprocessed
    print("\n[2/3] Validated algorithm (preprocessed)...")
    rows2 = []
    no_bouts_2 = 0
    for i, (_, row) in enumerate(home.iterrows()):
        c, sid, yr, d = row["cohort"], int(row["subj_id"]), int(row["year"]), int(row["sixmwd"])
        fname = f"{c}{sid:02d}_{yr}_{d}.csv"
        df = pd.read_csv(os.path.join(BASE, "csv_processed_home", fname))
        sig = df[["AP", "ML", "VT"]].values.astype(np.float32)
        bouts = validated_walking_detection(sig, amp_thresh=0.15)
        if bouts:
            walk = np.concatenate([sig[s:e] for s, e in bouts], axis=0)
        else:
            no_bouts_2 += 1
            walk = heuristic_walking_selection(sig)  # fallback
        rows2.append(extract_gait_features(walk))
        if (i + 1) % 50 == 0: print(f"  {i+1}/{n}", flush=True)
    print(f"  {n}/{n} (no bouts: {no_bouts_2})")

    # Method 3: Validated on raw csv_ca
    print("\n[3/3] Validated algorithm (raw csv_ca)...")
    rows3 = []
    no_bouts_3 = 0
    for i, (_, row) in enumerate(home.iterrows()):
        c, sid, yr, d = row["cohort"], int(row["subj_id"]), int(row["year"]), int(row["sixmwd"])
        fname = f"{c}{sid:02d}_{yr}_{d}.csv"
        fpath_ca = os.path.join(BASE, "csv_ca", fname)
        if os.path.exists(fpath_ca):
            df = pd.read_csv(fpath_ca)
            sig_raw = df[["X", "Y", "Z"]].values.astype(np.float32)
            bouts = validated_walking_detection_raw(sig_raw, amp_thresh=0.3)
            if bouts:
                walk_segs = [preprocess_segment(sig_raw[s:e]) for s, e in bouts[:10]]
                walk = np.concatenate(walk_segs, axis=0)
            else:
                no_bouts_3 += 1
                preprocessed = preprocess_segment(sig_raw)
                walk = heuristic_walking_selection(preprocessed)
        else:
            # Fallback to preprocessed
            df = pd.read_csv(os.path.join(BASE, "csv_processed_home", fname))
            sig = df[["AP", "ML", "VT"]].values.astype(np.float32)
            walk = heuristic_walking_selection(sig)
        rows3.append(extract_gait_features(walk))
        if (i + 1) % 50 == 0: print(f"  {i+1}/{n}", flush=True)
    print(f"  {n}/{n} (no bouts: {no_bouts_3})")

    # Build feature matrices
    def to_matrix(rows):
        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if df[col].isna().any(): df[col] = df[col].fillna(df[col].median())
        return df.values.astype(float)

    X1 = to_matrix(rows1)
    X2 = to_matrix(rows2)
    X3 = to_matrix(rows3)

    # LOO CV
    models = {
        "Ridge": lambda: Ridge(alpha=10.0),
        "ElasticNet": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        "SVR": lambda: SVR(kernel="rbf", C=100, epsilon=50),
    }

    def loo_featsel(X, y, mfn, top_k=15):
        preds = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            cors = np.array([abs(spearmanr(X[tr, j], y[tr])[0]) for j in range(X.shape[1])])
            idx = np.argsort(cors)[-top_k:]
            sc = StandardScaler()
            m = mfn()
            m.fit(sc.fit_transform(X[tr][:, idx]), y[tr])
            preds[te] = m.predict(sc.transform(X[te][:, idx]))
        return r2_score(y, preds)

    print(f"\n{'Model':15s} | {'Heuristic':>12s} | {'Validated(pp)':>14s} | {'Validated(raw)':>15s} | Best")
    print("-" * 75)
    for mname, mfn in models.items():
        r1 = loo_featsel(X1, y, mfn)
        r2 = loo_featsel(X2, y, mfn)
        r3 = loo_featsel(X3, y, mfn)
        best = max([(r1, "Heuristic"), (r2, "Val(pp)"), (r3, "Val(raw)")], key=lambda x: x[0])
        print(f"{mname:15s} | {r1:12.4f} | {r2:14.4f} | {r3:15.4f} | {best[1]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
