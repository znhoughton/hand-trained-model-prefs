#!/usr/bin/env python3
"""
Fix token counts in babylm-1.3b CSV files.

The tokens_per_step was incorrectly set to 1,638,400 (2x the correct value).
This script divides all token values by 2 for babylm-1.3b checkpoint files.
"""

import pandas as pd
from pathlib import Path
import sys

# UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def fix_babylm_1_3b_tokens():
    print("=" * 80)
    print("FIXING BABYLM-1.3B TOKEN COUNTS")
    print("=" * 80)

    # Find all babylm-1.3b CSV files
    data_dir = Path('../Data/checkpoint_results')
    pattern = 'opt-babylm-1.3b-64eps-seed964_step-*.csv'

    csv_files = sorted(data_dir.glob(pattern))

    if not csv_files:
        print(f"ERROR: No babylm-1.3b CSV files found in {data_dir}")
        return

    print(f"\nFound {len(csv_files)} babylm-1.3b checkpoint files")
    print(f"Will divide 'tokens' column by 2 in each file")
    print(f"Proceeding...\n")

    fixed_count = 0
    error_count = 0

    for csv_file in csv_files:
        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            # Check if 'tokens' column exists
            if 'tokens' not in df.columns:
                print(f"WARNING: 'tokens' column not found in {csv_file.name}")
                error_count += 1
                continue

            # Get original token value (should be same for all rows in checkpoint)
            original_tokens = df['tokens'].iloc[0] if len(df) > 0 else 0

            # Divide tokens by 2
            df['tokens'] = df['tokens'] / 2

            # Convert to int64 (needed for large token values)
            df['tokens'] = df['tokens'].astype('int64')

            # Write back
            df.to_csv(csv_file, index=False)

            new_tokens = df['tokens'].iloc[0] if len(df) > 0 else 0

            if fixed_count % 10 == 0:
                print(f"  [{fixed_count + 1:3d}/{len(csv_files)}] {csv_file.name}")
                print(f"            {original_tokens:,} → {new_tokens:,} tokens")

            fixed_count += 1

        except Exception as e:
            print(f"ERROR processing {csv_file.name}: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files processed successfully: {fixed_count}")
    print(f"Errors: {error_count}")
    print("\nDone! Token counts have been corrected.")


if __name__ == "__main__":
    fix_babylm_1_3b_tokens()
