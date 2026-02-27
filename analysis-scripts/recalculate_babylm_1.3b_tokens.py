#!/usr/bin/env python3
"""
Recalculate token counts in babylm-1.3b CSV files from step numbers.

The tokens_per_step was incorrectly set to 1,638,400.
The correct value is 819,200 (1024 × 100 × 4 × 2).

This script recalculates tokens = step × 819_200 for all babylm-1.3b files.
"""

import pandas as pd
from pathlib import Path
import sys
import re

# UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Correct tokens_per_step for babylm-1.3b
CORRECT_TOKENS_PER_STEP = 819_200


def recalculate_babylm_1_3b_tokens():
    print("=" * 80)
    print("RECALCULATING BABYLM-1.3B TOKEN COUNTS FROM STEP NUMBERS")
    print("=" * 80)
    print(f"Using tokens_per_step = {CORRECT_TOKENS_PER_STEP:,}")

    # Find all babylm-1.3b CSV files
    data_dir = Path('../Data/checkpoint_results')
    pattern = 'opt-babylm-1.3b-64eps-seed964_step-*.csv'

    csv_files = sorted(data_dir.glob(pattern))

    if not csv_files:
        print(f"ERROR: No babylm-1.3b CSV files found in {data_dir}")
        return

    print(f"\nFound {len(csv_files)} babylm-1.3b checkpoint files")
    print(f"Will recalculate 'tokens' from step number\n")

    fixed_count = 0
    error_count = 0

    for csv_file in csv_files:
        try:
            # Extract step number from filename
            match = re.search(r'step-(\d+)\.csv$', csv_file.name)
            if not match:
                print(f"WARNING: Could not extract step from {csv_file.name}")
                error_count += 1
                continue

            step = int(match.group(1))

            # Read CSV
            df = pd.read_csv(csv_file)

            # Check if columns exist
            if 'step' not in df.columns or 'tokens' not in df.columns:
                print(f"WARNING: Missing columns in {csv_file.name}")
                error_count += 1
                continue

            # Get original token value
            original_tokens = df['tokens'].iloc[0] if len(df) > 0 else 0

            # Recalculate tokens from step
            correct_tokens = step * CORRECT_TOKENS_PER_STEP

            # Update all rows
            df['tokens'] = correct_tokens

            # Verify step column matches filename
            if (df['step'] != step).any():
                print(f"WARNING: step column mismatch in {csv_file.name}")

            # Write back
            df.to_csv(csv_file, index=False)

            if fixed_count % 10 == 0:
                print(f"  [{fixed_count + 1:3d}/{len(csv_files)}] {csv_file.name}")
                print(f"            Step {step:,}: {original_tokens:,} → {correct_tokens:,} tokens")

            fixed_count += 1

        except Exception as e:
            print(f"ERROR processing {csv_file.name}: {e}")
            error_count += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files processed successfully: {fixed_count}")
    print(f"Errors: {error_count}")
    print("\nDone! Token counts have been recalculated correctly.")


if __name__ == "__main__":
    recalculate_babylm_1_3b_tokens()
