#!/usr/bin/env python3
"""
Compare: Activity Index pre-filtering before walking bout detection.
Segment raw home signal by AI level, keep moderate-AI segments,
then apply walking bout detection on those segments only.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import spearmanr
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
FS = 30.0


def load_raw_signal(cohort, sid, year, sixmwd):
    fname = f"{cohort}{sid:02d}_{year}_{sixmwd}.csv"
    for source in ["csv_ca", "csv_processed_home"]:
        fpath = os.path.join(BASE, source, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if "X" in df.columns:
                return df[["X", "Y", "Z"]].values.astype(np.float32), "raw"
            return df[["AP", "ML", "VT"]].values.astype(np.float32), "preprocessed"
    raise FileNotFoundError(fname)


# ── Activity Index segmentation ──

def compute_activity_index_bins(sig, fs=FS, bin_sec=1.0):
    """Compute per-second Activity Index (RMS of VM)."""
    vm = np.sqrt(sig[:, 0]**2 + sig[:, 1]**2 + sig[:, 2]**2)
    bs = int(bin_sec * fs)
    n_bins = len(vm) // bs
    ai = np.array([np.sqrt(np.mean(vm[i*bs:(i+1)*bs]**2)) for i in range(n_bins)])
    return ai


def segment_by_activity_index(sig, fs=FS, bin_sec=1.0):
    """Segment signal into moderate-AI regions.
    Walking AI is moderate: above sedentary, below vigorous.
    For raw accel (g units): sedentary VM~1g (still), walking VM~1.0-1.3g.
    We use percentile-based thresholds adaptive to each subject.
    """
    ai = compute_activity_index_bins(sig, fs, bin_sec)
    bs = int(bin_sec * fs)

    # Adaptive thresholds: moderate = between 30th and 85th percentile
    # Low = rest/sedentary, High = vigorous/artifacts
    low_thresh = np.percentile(ai, 30)
    high_thresh = np.percentile(ai, 85)

    # Find continuous moderate-AI regions (min 5 seconds)
    moderate = (ai >= low_thresh) & (ai <= high_thresh)
    segments = []
    in_seg = False
    start = 0
    min_bins = 5  # 5 seconds minimum

    for i in range(len(moderate)):
        if moderate[i] and not in_seg:
            start = i
            in_seg = True
        elif not moderate[i] and in_seg:
            if (i - start) >= min_bins:
                segments.append((start * bs, i * bs))
            in_seg = False
    if in_seg and (len(moderate) - start) >= min_bins:
        segments.append((start * bs, len(moderate) * bs))

    return segments, ai, low_thresh, high_thresh


# ── Walking bout detection (same as main script) ──

def detect_walking_bouts_raw(sig_xyz, fs=FS, win_sec=10, step_sec=2, min_bout_sec=30):
    vm = np.sqrt(sig_xyz[:, 0]**2 + sig_xyz[:, 1]**2 + sig_xyz[:, 2]**2)
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    is_walking, starts = [], []
    for s in range(0, len(vm) - win, step):
        seg_vm = vm[s:s + win]
        rms = np.sqrt(np.mean(seg_vm**2))
        std = np.std(seg_vm)
        seg = sig_xyz[s:s + win]
        variances = [np.var(seg[:, i]) for i in range(3)]
        best_axis = seg[:, np.argmax(variances)]
        best_c = best_axis - best_axis.mean()
        acf = np.correlate(best_c, best_c, "full")[len(best_c) - 1:]
        acf /= (acf[0] + 1e-12)
        search = acf[int(0.3 * fs):int(1.5 * fs)]
        peaks, props = find_peaks(search, height=0.1)
        regularity = props["peak_heights"][0] if len(peaks) > 0 else 0
        walking = (std > 0.05) and (rms > 0.8) and (rms < 1.5) and (regularity > 0.15)
        is_walking.append(walking)
        starts.append(s)

    bouts = []
    in_bout = False
    bout_start = 0
    for i, (w, s) in enumerate(zip(is_walking, starts)):
        if w and not in_bout:
            bout_start = s
            in_bout = True
        elif not w and in_bout:
            bout_end = starts[i - 1] + win
            if (bout_end - bout_start) >= min_bout_sec * fs:
                bouts.append((bout_start, bout_end))
            in_bout = False
    if in_bout:
        bout_end = starts[-1] + win
        if (bout_end - bout_start) >= min_bout_sec * fs:
            bouts.append((bout_start, bout_end))
    return bouts


def preprocess_walking_segment(seg_xyz, fs=FS):
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


def score_windows_preprocessed(sig, fs=FS, win_sec=10, step_sec=2):
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
    return np.array(scores), np.array(starts)


def select_best_walking_preprocessed(sig, top_pct=0.25, fs=FS):
    scores, starts = score_windows_preprocessed(sig, fs)
    win = int(10 * fs)
    n_sel = max(5, int(len(scores) * top_pct))
    n_sel = min(n_sel, len(scores))
    top_idx = np.sort(np.argsort(scores)[-n_sel:])
    return np.concatenate([sig[starts[i]:starts[i] + win] for i in top_idx])


# ── Gait feature extraction (same 35 features) ──

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


# ── Pipeline: AI filter → walking detection → preprocess → features ──

def pipeline_ai_filtered(cohort, sid, year, sixmwd):
    """AI pre-filter → walking detection → preprocess → features."""
    sig, sig_type = load_raw_signal(cohort, sid, year, sixmwd)

    if sig_type == "raw":
        # Step 1: Segment by Activity Index → keep moderate
        mod_segs, ai, lo, hi = segment_by_activity_index(sig)

        if mod_segs:
            # Concatenate moderate-AI segments
            mod_signal = np.concatenate([sig[s:e] for s, e in mod_segs], axis=0)
        else:
            mod_signal = sig  # fallback: use all

        # Step 2: Detect walking bouts within moderate-AI signal
        bouts = detect_walking_bouts_raw(mod_signal)

        if bouts:
            walk_segs = [preprocess_walking_segment(mod_signal[s:e])
                         for s, e in bouts[:10]]
            walk = np.concatenate(walk_segs, axis=0)
        else:
            # Fallback: preprocess moderate signal, use window selection
            preprocessed = preprocess_walking_segment(mod_signal)
            walk = select_best_walking_preprocessed(preprocessed)
    else:
        # Already preprocessed — use window selection
        walk = select_best_walking_preprocessed(sig)

    return extract_gait_features(walk)


def pipeline_no_ai_filter(cohort, sid, year, sixmwd):
    """Direct walking detection on raw (no AI pre-filter) — current best."""
    sig, sig_type = load_raw_signal(cohort, sid, year, sixmwd)

    if sig_type == "raw":
        bouts = detect_walking_bouts_raw(sig)
        if bouts:
            walk_segs = [preprocess_walking_segment(sig[s:e])
                         for s, e in bouts[:10]]
            walk = np.concatenate(walk_segs, axis=0)
        else:
            preprocessed = preprocess_walking_segment(sig)
            walk = select_best_walking_preprocessed(preprocessed)
    else:
        walk = select_best_walking_preprocessed(sig)

    return extract_gait_features(walk)


# ── Main comparison ──

def main():
    print("=" * 70)
    print("Comparing: AI pre-filter vs direct walking detection")
    print("=" * 70)

    home = pd.read_csv(os.path.join(BASE, "sway_features_home.csv"))
    home.rename(columns={"year_x": "year"}, inplace=True)
    y = home["sixmwd"].values.astype(float)
    n = len(y)

    # Extract features with both pipelines
    print("\nExtracting features — AI filtered...")
    ai_rows = []
    for i, (_, row) in enumerate(home.iterrows()):
        ai_rows.append(pipeline_ai_filtered(
            row["cohort"], int(row["subj_id"]), int(row["year"]), int(row["sixmwd"])))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}", flush=True)
    print(f"  {n}/{n}")

    print("Extracting features — no AI filter (current best)...")
    noai_rows = []
    for i, (_, row) in enumerate(home.iterrows()):
        noai_rows.append(pipeline_no_ai_filter(
            row["cohort"], int(row["subj_id"]), int(row["year"]), int(row["sixmwd"])))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}", flush=True)
    print(f"  {n}/{n}")

    def to_matrix(rows):
        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        for c in df.columns:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
        return df.values.astype(float)

    X_ai = to_matrix(ai_rows)
    X_noai = to_matrix(noai_rows)

    # LOO CV comparison (A1 config: feature selection, top_k=15)
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

    print(f"\n{'Model':15s} | {'No AI filter':>14s} | {'AI pre-filter':>14s} | Winner")
    print("-" * 60)
    for mname, mfn in models.items():
        r2_noai = loo_featsel(X_noai, y, mfn)
        r2_ai = loo_featsel(X_ai, y, mfn)
        winner = "AI-FILTER" if r2_ai > r2_noai else "NO-FILTER"
        print(f"{mname:15s} | {r2_noai:14.4f} | {r2_ai:14.4f} | {winner}")

    print("\nDone.")


if __name__ == "__main__":
    main()
