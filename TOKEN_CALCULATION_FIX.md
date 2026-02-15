# Token Calculation Fix - Summary

## Problem Identified

The `tokens_per_step` value for BabyLM-125M in `model-prefs-all-ckpts.py` was incorrect, causing all token counts to be inflated by ~40%.

## Root Cause

**Incorrect value in analysis script**:
```python
"znhoughton/opt-babylm-125m-100eps-seed964": {
    "tokens_per_step": 819_200,  # WRONG
}
```

**Correct calculation**:
- Dataset: 100M tokens
- Epochs: 100
- Total tokens: 10,000,000,000 (10B)
- Actual training steps: 17,100
- **Correct tokens_per_step**: 10,000,000,000 ÷ 17,100 = **584,795**

## Changes Made

### 1. Fixed `analysis-scripts/model-prefs-all-ckpts.py`
- Line 108: Changed `tokens_per_step` from 819,200 to 584,795
- Updated comment to reflect actual calculation

### 2. Fixed `RESEARCH_DOCUMENTATION.md`
- Updated OPT-125M training parameters table (tokens per step, save steps, total steps)
- Changed "14B tokens total" to "10B tokens total" 
- Removed incorrect dataset note about 140M tokens
- Fixed "~140×" to "100×" (correct number of epochs)
- Changed "10-14B" to "10B" in training dynamics discussion

## Impact on Existing Data

**ALL existing checkpoint CSV files** in `Data/checkpoint_results/` still contain inflated token counts because they were generated with the old (incorrect) `tokens_per_step` value.

### Correction Factor
To convert old token values to correct values: **multiply by 0.7140**
- Old: 14.0B tokens → Correct: 10.0B tokens  
- Old: 7.0B tokens → Correct: 5.0B tokens

## Next Steps

You have **two options**:

### Option 1: Re-run Analysis (Time-consuming but accurate)
```bash
cd analysis-scripts
python model-prefs-all-ckpts.py  # Re-evaluate all checkpoints
Rscript -e "rmarkdown::render('analysis-script.Rmd')"  # Re-generate plots
```

This will:
- Regenerate all CSV files with correct token counts
- Create new plots with correct x-axis labels
- Ensure all data is consistent

### Option 2: Apply Correction Factor to Plots (Quick fix)
The R analysis script reads the CSV files which have old token values. You can either:
1. Manually note in presentations that x-axis should be multiplied by 0.714
2. Post-process plots to relabel x-axis
3. Use the corrected documentation (which now has correct values)

## Verification

✅ Training script confirms: 100 epochs × 100M tokens = 10B total
✅ Final step: 17,100 (confirmed from checkpoint data)  
✅ Correct calculation: 10B ÷ 17,100 = 584,795 tokens/step
✅ Documentation updated with correct values
✅ Analysis script fixed for future runs

---

**Date**: 2026-02-13
**Issue**: Token calculation error in BabyLM-125M analysis
**Status**: Fixed in code and documentation; CSV data requires regeneration
