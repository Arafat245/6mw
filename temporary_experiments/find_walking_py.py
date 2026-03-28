#!/usr/bin/env python3
"""
Python port of find_walking.m (Straczkiewicz et al., npj Digital Medicine, 2023).
"A one-size-fits-most walking recognition method for smartphones, smartwatches,
and wearable accelerometers."

Original MATLAB: https://github.com/MStraczkiewicz/find_walking

Parameters for waist/hip accelerometer:
  min_amp = 0.3 g (amplitude threshold)
  step_freq = [1.4, 2.3] Hz
  alpha = 0.6 (ratio below step freq)
  beta = 2.5 (ratio above step freq)
  T = 3 (minimum walking duration in seconds)
  delta = 2 (max cadence drift between consecutive seconds, in 0.05Hz bins)
"""
import numpy as np
import pywt
from scipy.signal import find_peaks, resample
from scipy.interpolate import RectBivariateSpline


def _peak2peak_per_second(vm, fs):
    """Peak-to-peak amplitude per 1-second window."""
    n_sec = len(vm) // fs
    vm_trimmed = vm[:n_sec * fs].reshape(n_sec, fs)
    return np.ptp(vm_trimmed, axis=1)


def _peakseek(x):
    """Simple peak finding: local maxima."""
    peaks = []
    vals = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
            vals.append(x[i])
    return np.array(peaks, dtype=int), np.array(vals)


def find_continuous_dominant_peaks(W, T, delta):
    """
    Identify consecutive peaks in W that occur for at least T consecutive
    columns and whose row difference is no more than delta.

    W: (n_freqs, n_seconds) binary matrix with peak locations
    T: minimum consecutive seconds
    delta: max row index drift between consecutive peaks
    Returns: B, same shape as W, with validated peaks
    """
    n_rows, n_cols = W.shape
    B = np.zeros_like(W)

    for m in range(n_cols - T + 1):
        A = W[:, m:m+T].copy()

        loop = list(range(T)) + list(range(T-2, -1, -1))
        broken = False

        for t_idx in range(len(loop)):
            s = loop[t_idx]
            Pr = np.where(A[:, s] > 0)[0]
            j = 0

            for pk in Pr:
                neighbors = np.arange(max(0, pk - delta), min(n_rows, pk + delta + 1))

                if s == 0:
                    # First column: check right neighbor
                    has_right = np.any(A[neighbors, min(s+1, T-1)] > 0)
                    if not has_right:
                        A[pk, s] = 0
                    else:
                        j += 1
                elif s == T - 1:
                    # Last column: check left neighbor
                    has_left = np.any(A[neighbors, s-1] > 0)
                    if not has_left:
                        A[pk, s] = 0
                    else:
                        j += 1
                else:
                    # Middle: check both neighbors
                    has_left = np.any(A[neighbors, s-1] > 0)
                    has_right = np.any(A[neighbors, s+1] > 0)
                    if not (has_left and has_right):
                        A[pk, s] = 0
                    else:
                        j += 1

            if j == 0:
                A[:] = 0
                broken = True
                break

        if not broken:
            B[:, m:m+T] = np.maximum(B[:, m:m+T], A)

    return B


def find_walking(vm, fs, min_amp=0.3, T=3, delta=2, alpha=0.6, beta=2.5,
                 step_freq=(1.4, 2.3)):
    """
    Identify walking periods from vector magnitude of acceleration.

    Parameters:
        vm: 1D array, vector magnitude of raw acceleration
        fs: int, sampling frequency (Hz)
        min_amp: float, minimum peak-to-peak amplitude per second (g)
        T: int, minimum walking duration (seconds)
        delta: int, max cadence drift (in 0.05Hz bins)
        alpha: float, ratio threshold for below step_freq
        beta: float, ratio threshold for above step_freq
        step_freq: tuple, walking cadence range (Hz)

    Returns:
        wi: 1D boolean array per sample (same length as vm), True = walking
        steps: float, total estimated steps
        cad_per_sec: 1D array per second, cadence (Hz)
    """
    vm = np.asarray(vm, dtype=float).ravel()
    n_samples = len(vm)
    n_sec = n_samples // fs

    # Preallocate
    wi = np.zeros(n_samples, dtype=bool)
    cad_per_sec = np.zeros(n_sec)

    if n_sec < T:
        return wi, 0.0, cad_per_sec

    # Peak-to-peak amplitude per second
    pp = _peak2peak_per_second(vm, fs)
    valid = pp >= min_amp  # boolean per second

    # Expand valid to sample level and trim VM
    valid_samples = np.repeat(valid, fs)[:n_samples]
    vm_valid = vm[valid_samples[:len(vm)]]

    if len(vm_valid) < T * fs:
        return wi, 0.0, cad_per_sec

    vm_len = len(vm_valid)

    # Tukey window to smooth edges
    from scipy.signal.windows import tukey
    w = tukey(vm_len, 0.02)
    vm_windowed = vm_valid * w

    # Pad with zeros (5 seconds each side)
    pad = np.zeros(5 * fs)
    vm_padded = np.concatenate([pad, vm_windowed, pad])

    # CWT using pywt — Morlet wavelet
    freq_min = 0.5
    freq_max = min(fs / 2, 10.0)
    freqs_target = np.arange(round(freq_min, 1), round(freq_max, 1) + 0.05, 0.05)

    # For pywt cmor wavelet: scale = central_freq * fs / freq
    # Using cmor1.5-1.0 (bandwidth=1.5, center_freq=1.0)
    central_freq = pywt.central_frequency('cmor1.5-1.0')
    scales = central_freq * fs / freqs_target

    # Handle short signals by repeating
    if len(vm_padded) < 50 * fs:
        vm_for_cwt = np.tile(vm_padded, 10)
    else:
        vm_for_cwt = vm_padded

    # CWT
    coeffs, cwt_freqs = pywt.cwt(vm_for_cwt, scales, 'cmor1.5-1.0',
                                   sampling_period=1.0/fs)
    Cabs = np.abs(coeffs) ** 2

    # Trim padding
    start_idx = 5 * fs
    if len(vm_padded) < 50 * fs:
        Cabs = Cabs[:, start_idx:start_idx + vm_len]
    else:
        Cabs = Cabs[:, start_idx:start_idx + vm_len]

    if Cabs.shape[1] < fs:
        return wi, 0.0, cad_per_sec

    # Frequency grid (already linear at 0.05 Hz steps)
    freqs_linspace = freqs_target

    # Get location of step frequency boundaries
    loc1 = np.argmin(np.abs(freqs_linspace - step_freq[0]))
    loc2 = np.argmin(np.abs(freqs_linspace - step_freq[1]))

    # Per-second analysis
    n_valid_sec = Cabs.shape[1] // fs
    D = np.zeros((len(freqs_linspace), n_valid_sec))

    for i in range(n_valid_sec):
        seg_start = i * fs
        seg_end = seg_start + fs
        if seg_end > Cabs.shape[1]:
            break

        # Sum CWT power over this second
        power_1s = np.sum(Cabs[:, seg_start:seg_end], axis=1)

        # Find peaks
        pks_locs, pks_vals = _peakseek(power_1s)
        if len(pks_locs) == 0:
            continue

        # Sort by peak value (descending)
        sort_idx = np.argsort(pks_vals)[::-1]
        pks_locs = pks_locs[sort_idx]
        pks_vals = pks_vals[sort_idx]

        # Check if any peak is within step frequency range
        step_mask = (pks_locs >= loc1) & (pks_locs <= loc2)
        step_pks = np.where(step_mask)[0]
        if len(step_pks) == 0:
            continue

        first_step_pk = step_pks[0]  # highest peak within step freq

        if pks_locs[0] > loc2:
            # Highest peak above step freq (could be running)
            if pks_vals[0] / (pks_vals[first_step_pk] + 1e-12) < beta:
                D[pks_locs[first_step_pk], i] = 1
        elif pks_locs[0] < loc1:
            # Highest peak below step freq (could be arm swing)
            if pks_vals[0] / (pks_vals[first_step_pk] + 1e-12) < alpha:
                D[pks_locs[first_step_pk], i] = 1
        else:
            # Highest peak IS within step freq range
            D[pks_locs[first_step_pk], i] = 1

    # Align peaks with valid seconds
    E = np.zeros((D.shape[0], n_sec))
    valid_indices = np.where(valid[:n_sec])[0]
    if len(valid_indices) <= D.shape[1]:
        E[:, valid_indices[:D.shape[1]]] = D[:, :len(valid_indices)]
    else:
        E[:, valid_indices[:D.shape[1]]] = D

    # Check periodicity
    if T > 1:
        B = find_continuous_dominant_peaks(E, T, delta)
    else:
        B = E

    # Walking indicator per second
    e = np.sum(B, axis=0)
    wi_sec = (e > 0).astype(int)

    # Expand to sample level
    wi = np.repeat(wi_sec, fs)[:n_samples]
    wi = wi.astype(bool)

    # Estimate cadence per second
    for i in range(min(n_sec, B.shape[1])):
        ind_freqs = np.where(B[:, i] > 0)[0]
        if len(ind_freqs) == 1:
            cad_per_sec[i] = freqs_linspace[ind_freqs[0]]

    steps = float(np.sum(cad_per_sec))

    return wi, steps, cad_per_sec


def detect_walking_bouts_findwalking(xyz, fs, min_amp=0.3, T=3, delta=2,
                                      alpha=0.6, beta=2.5,
                                      step_freq=(1.4, 2.3),
                                      min_bout_sec=10, merge_gap_sec=5):
    """
    Wrapper: run find_walking on XYZ data, return list of (start_sample, end_sample) bouts.
    """
    vm = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)

    wi, steps, cad = find_walking(vm, fs, min_amp, T, delta, alpha, beta, step_freq)

    # Convert per-sample boolean to bout list
    wi_sec = wi[::fs] if len(wi) >= fs else wi  # per-second
    n_sec = len(wi_sec)

    # Find contiguous walking segments
    bouts = []
    in_bout = False
    bout_start = 0

    for s in range(n_sec):
        if wi_sec[s] and not in_bout:
            bout_start = s
            in_bout = True
        elif not wi_sec[s] and in_bout:
            # Check gap — merge if small
            gap = 0
            for g in range(s, min(s + merge_gap_sec + 1, n_sec)):
                if wi_sec[g]:
                    gap = -1  # gap filled
                    break
                gap += 1
            if gap <= merge_gap_sec and gap >= 0:
                continue  # keep in bout (gap will be filled)
            # End bout
            if s - bout_start >= min_bout_sec:
                bouts.append((bout_start * fs, s * fs))
            in_bout = False

    if in_bout and n_sec - bout_start >= min_bout_sec:
        bouts.append((bout_start * fs, n_sec * fs))

    return bouts


if __name__ == '__main__':
    # Quick test
    import time
    import pandas as pd
    from pathlib import Path

    BASE = Path(__file__).parent.parent
    fp = BASE / 'csv_home_daytime' / 'C01_2016_2147.csv'
    xyz = pd.read_csv(fp, usecols=['X', 'Y', 'Z']).values.astype(float)

    print(f"Testing find_walking on C01 ({len(xyz)} samples, {len(xyz)/30:.0f}s)...")
    t0 = time.time()
    bouts = detect_walking_bouts_findwalking(xyz, 30)
    elapsed = time.time() - t0

    total_walk = sum((e-s)/30 for s, e in bouts)
    print(f"  {len(bouts)} bouts, total walking: {total_walk/60:.1f} min")
    print(f"  Longest bout: {max((e-s)/30 for s, e in bouts):.0f}s" if bouts else "  No bouts")
    print(f"  Time: {elapsed:.1f}s")
