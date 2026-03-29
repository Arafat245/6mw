#!/usr/bin/env python3
"""
Verify walking bouts using walking_verify.py heuristics.
Copies only verified bouts to a new folder.

Input:  walking_bouts/{subject_id}/bout_*.csv (Timestamp, X, Y, Z)
Output: verified_walking_bouts/{subject_id}/bout_*.csv (only verified bouts)

Run:  python temporary_experiments/verify_walking_bouts.py [--bout-dir walking_bouts] [--out-dir verified_walking_bouts]
"""
import argparse, time, shutil
import pandas as pd
from pathlib import Path
import sys

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'notebooks'))
from walking_verify import verify_walking_segment_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bout-dir', default='walking_bouts', help='Input bout folder')
    parser.add_argument('--out-dir', default='verified_walking_bouts', help='Output verified bout folder')
    args = parser.parse_args()

    t0 = time.time()
    BOUT_DIR = BASE / args.bout_dir
    OUT_DIR = BASE / args.out_dir

    subj_dirs = sorted([d for d in BOUT_DIR.iterdir() if d.is_dir()])
    n = len(subj_dirs)
    print(f"Verifying walking bouts for {n} subjects...")
    print(f"  Input:  {args.bout_dir}/")
    print(f"  Output: {args.out_dir}/")

    total_in = 0
    total_verified = 0
    total_rejected = 0

    for i, subj_dir in enumerate(subj_dirs):
        subj_key = subj_dir.name
        bout_files = sorted(subj_dir.glob('bout_*.csv'))
        n_verified = 0

        out_subj_dir = OUT_DIR / subj_key
        out_subj_dir.mkdir(parents=True, exist_ok=True)

        for bf_path in bout_files:
            total_in += 1
            df = pd.read_csv(bf_path)

            result_df = verify_walking_segment_df(df)
            # Extract is_walking from the result DataFrame
            is_walking = result_df.loc[result_df['metric'] == 'is_walking', 'value'].iloc[0]

            if is_walking:
                shutil.copy2(bf_path, out_subj_dir / bf_path.name)
                n_verified += 1
                total_verified += 1
            else:
                total_rejected += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}] {subj_key}: {n_verified}/{len(bout_files)} verified", flush=True)

    print(f"\nTotal: {total_verified}/{total_in} bouts verified ({total_rejected} rejected)")
    print(f"Rejection rate: {100*total_rejected/total_in:.1f}%")
    print(f"Saved to {args.out_dir}/")
    print(f"Done in {time.time()-t0:.0f}s")
