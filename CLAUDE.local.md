# Session Memory — 2026-03-28 to 2026-03-29

## Key Findings

### Sampling Frequencies
- **91 subjects at 30 Hz, 8 at 60 Hz (M26, M27, M29, M31, M32, M43, M45, C75), 2 at 100 Hz (M48, M49)**
- M44 (60 Hz) is excluded from analysis
- Sampling rate is in GT3X metadata (`info.txt` inside ZIP, field `Sample Rate`)
- **Hardcoded FS=30 for all subjects gives best results** — resampling (resample_poly or linear interp) hurts performance
- Resampled results: resample_poly R²=0.342, linear interp R²=0.373, hardcoded 30 Hz R²=0.454

### Full Recording vs Daytime Filtering
- **Using full recording (no daytime filter) gives R²=0.454**
- Daytime only (7-22h) gives R²=0.365
- Worn-time filter only gives R²=0.407
- No filter at all = best, because sedentary/sleep patterns are informative for 6MWD prediction
- The worn-time filter (`rolling 5s std of VM > 0.01`) removes real sedentary data, not just non-wear

### Walking Bout Verification
- `notebooks/walking_verify.py` has heuristic bout verification
- Verified bouts (42% rejected) give R²=0.423 — worse than keeping all bouts (R²=0.454)
- Non-walking bouts contain useful movement patterns that help prediction

### pygt3x Behavior
- pygt3x 0.7.1 returns `float64` timestamps (Unix epoch seconds) on both Linux and Windows
- `.NET ticks` in GT3X metadata represent **local time** — so `ts % 86400 / 3600` gives local hours
- The `astype(np.int64) // 10**9` conversion in windows_cursor branch was a bug that accidentally disabled daytime filtering, which improved results

### Clinic Per-Bout Features
- Splitting 6MWT into fixed-duration windows: **60s windows better than 30s**
- 60s PerBout-Top20+Demo(4): R²=0.675, MAE=132
- Doesn't beat Gait+CWT+WS+Demo (R²=0.806) but useful for clinic-home comparison

### File Matching
- `target_6mwd.csv` has updated years that don't match `csv_raw2/` and `csv_preprocessed2/` filenames
- Must match by `cohort + subj_id` (e.g., `C61`) not full filename (e.g., `C61_2019_2111.csv` vs `C61_2017_2111.csv`)

## Current Best Results

| Setting | Features | R² | MAE (ft) | ρ |
|---------|----------|-----|---------|---|
| **Clinic** | Gait+CWT+WS+Demo (55f) | **0.806** | **102** | **0.880** |
| **Home** | PerBout-Top20+Demo(4) (24f) | **0.454** | **182** | **0.659** |

## Pipeline (Home)

```
python home/step0_gt3x_to_npz.py          # GT3X → NPZ (~60 min, one-time)
python home/step1_detect_walking_bouts.py --save-csv  # Detect + save bouts (~18 min)
python home/step2_extract_features.py      # Extract 153 features (~12 min)
python home/step3_predict.py               # LOO CV → R²=0.454 (<1 sec)
```

Quick reproduction: `python home/step3_predict.py` (uses cached features)

Full results: `python analysis/results_table_final.py` (clinic + home)

## Preferences
- 60s windows for clinic per-bout features
- No daytime filtering for home data
- No resampling — use hardcoded 30 Hz
- Keep all walking bouts (don't verify/filter)
- Save walking bouts as CSV with Timestamp, X, Y, Z
- Option `--save-csv` for bout saving, `--bout-dir` for custom folder
