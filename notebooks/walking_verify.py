from scipy.signal import butter, filtfilt, welch, find_peaks
import numpy as np
import pandas as pd

_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz


def _verify_walking_metrics_to_df(result: dict) -> pd.DataFrame:
    """Always return metric / value rows so callers get a stable schema."""
    return (
        pd.DataFrame.from_dict(result, orient="index", columns=["value"])
        .reset_index()
        .rename(columns={"index": "metric"})
    )


def verify_walking_segment_df(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    xyz_cols: tuple[str, str, str] = ("X", "Y", "Z"),
    band_hz: tuple[float, float] = (0.8, 3.0),
    min_duration_s: float = 5.0,
    min_peak_count: int = 6,
    min_peak_prominence: float = 0.03,
    peak_distance_s: float = 0.30,
    min_band_power_ratio: float = 0.35,
    min_peak_sharpness_ratio: float = 1.8,
    require_cadence_agreement: bool = False,
    max_cadence_rel_diff: float = 0.35,
    cadence_spm_bounds: tuple[float, float] = (40.0, 180.0),
) -> pd.DataFrame:
    """
    Heuristic check for gait-like periodicity in tri-axial accel (Timestamp + X,Y,Z).

    `cadence_steps_per_min` uses PSD peak frequency × 60 (interpretation depends on sensor).
    `cadence_from_peaks_spm` uses median time between filtered magnitude peaks—useful cross-check.
    """
    xcol, ycol, zcol = xyz_cols
    d = df[[timestamp_col, xcol, ycol, zcol]].dropna().reset_index(drop=True)

    lo_cad, hi_cad = cadence_spm_bounds

    def base_result() -> dict:
        return {
            "is_walking": False,
            "n_samples": int(len(d)),
            "fs_est": np.nan,
            "duration_s": np.nan,
            "dominant_freq_hz": np.nan,
            "cadence_steps_per_min": np.nan,
            "cadence_from_peaks_spm": np.nan,
            "median_inter_peak_s": np.nan,
            "cadence_relative_diff": np.nan,
            "n_peaks": 0,
            "band_power_ratio": np.nan,
            "peak_sharpness_ratio": np.nan,
            "reasons": "",
        }

    result = base_result()

    if len(d) < 3:
        result["reasons"] = "too few samples"
        return _verify_walking_metrics_to_df(result)

    t = d[timestamp_col].to_numpy(dtype=float)
    x = d[xcol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)
    z = d[zcol].to_numpy(dtype=float)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        result["reasons"] = "could not estimate fs"
        return _verify_walking_metrics_to_df(result)

    fs_est = float(1.0 / np.median(dt))
    duration_s = float(t[-1] - t[0])

    result["fs_est"] = fs_est
    result["duration_s"] = duration_s

    if duration_s < min_duration_s:
        result["reasons"] = "segment too short"
        return _verify_walking_metrics_to_df(result)

    mag = np.sqrt(x * x + y * y + z * z)
    mag = mag - np.median(mag)

    low, high = band_hz
    nyq = 0.5 * fs_est
    low_n = low / nyq
    high_n = high / nyq

    if not (0 < low_n < high_n < 1):
        result["reasons"] = "invalid band for nyquist"
        return _verify_walking_metrics_to_df(result)

    b, a = butter(4, [low_n, high_n], btype="bandpass")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(mag) <= padlen:
        result["reasons"] = "segment too short for filtfilt pad"
        return _verify_walking_metrics_to_df(result)

    sig = filtfilt(b, a, mag)

    nperseg = min(len(sig), max(int(fs_est * 4), 32))
    freqs, psd = welch(sig, fs=fs_est, nperseg=nperseg, detrend="constant")

    band_mask = (freqs >= low) & (freqs <= high)
    if not np.any(band_mask):
        result["reasons"] = "no spectral bins in band"
        return _verify_walking_metrics_to_df(result)

    total_power = float(_TRAPEZOID(psd, freqs))
    band_power = float(_TRAPEZOID(psd[band_mask], freqs[band_mask]))
    band_power_ratio = band_power / total_power if total_power > 0 else 0.0

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]
    dominant_freq_hz = float(band_freqs[np.argmax(band_psd)])
    med_band = float(np.median(band_psd))
    peak_sharpness_ratio = (
        float(np.max(band_psd) / med_band) if med_band > 0 else float("inf")
    )

    peaks, _ = find_peaks(
        sig,
        distance=max(1, int(fs_est * peak_distance_s)),
        prominence=min_peak_prominence,
    )

    cadence_steps_per_min = float(dominant_freq_hz * 60.0)

    cadence_from_peaks_spm = float("nan")
    median_inter_peak_s = float("nan")
    if len(peaks) >= 2:
        peak_times = t[peaks]
        ipe = np.diff(peak_times)
        ipe = ipe[np.isfinite(ipe) & (ipe > 0)]
        if len(ipe):
            median_inter_peak_s = float(np.median(ipe))
            if median_inter_peak_s > 0:
                cadence_from_peaks_spm = float(60.0 / median_inter_peak_s)

    # Relative difference between PSD- and peak-based cadence (diagnostic / optional gate)
    cadence_relative_diff = float("nan")
    if np.isfinite(cadence_steps_per_min) and np.isfinite(cadence_from_peaks_spm):
        denom = max(abs(cadence_steps_per_min), abs(cadence_from_peaks_spm), 1.0)
        cadence_relative_diff = float(
            abs(cadence_steps_per_min - cadence_from_peaks_spm) / denom
        )

    reasons: list[str] = []
    ok = True

    if band_power_ratio < min_band_power_ratio:
        ok = False
        reasons.append("low band power ratio")

    if peak_sharpness_ratio < min_peak_sharpness_ratio:
        ok = False
        reasons.append("weak dominant spectral peak")

    if len(peaks) < min_peak_count:
        ok = False
        reasons.append("too few peaks")

    if not (lo_cad <= cadence_steps_per_min <= hi_cad):
        ok = False
        reasons.append("PSD cadence out of range")

    if require_cadence_agreement:
        if len(peaks) < 2 or not np.isfinite(cadence_from_peaks_spm):
            ok = False
            reasons.append("cannot verify peak cadence")
        elif not (lo_cad <= cadence_from_peaks_spm <= hi_cad):
            ok = False
            reasons.append("peak cadence out of range")
        elif not np.isfinite(cadence_relative_diff) or (
            cadence_relative_diff > max_cadence_rel_diff
        ):
            ok = False
            reasons.append("PSD vs peak cadence disagree")

    if ok:
        reasons.append("stable periodic motion consistent with walking")

    result.update(
        {
            "is_walking": ok,
            "dominant_freq_hz": dominant_freq_hz,
            "cadence_steps_per_min": cadence_steps_per_min,
            "cadence_from_peaks_spm": cadence_from_peaks_spm,
            "median_inter_peak_s": median_inter_peak_s,
            "cadence_relative_diff": cadence_relative_diff,
            "n_peaks": int(len(peaks)),
            "band_power_ratio": band_power_ratio,
            "peak_sharpness_ratio": peak_sharpness_ratio,
            "reasons": "; ".join(reasons),
        }
    )

    return _verify_walking_metrics_to_df(result)