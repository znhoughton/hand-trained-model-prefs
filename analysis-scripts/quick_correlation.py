#!/usr/bin/env python3
"""Quick correlation check - samples every 10th checkpoint"""
import pandas as pd
import glob
import sys

# UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load human data
human_data = pd.read_csv('../Data/all_human_data.csv')
human_data['Attested'] = (human_data['OverallFreq'] > 0).astype(int)
human_data['resp_binary'] = human_data['resp'].map({'alpha': 1, 'nonalpha': 0})
human_data['binom'] = human_data['Alpha']

avg_human_pref = human_data.groupby('binom').agg({
    'resp_binary': 'mean',
    'Attested': 'first'
}).reset_index()
avg_human_pref.rename(columns={'resp_binary': 'hum_pref'}, inplace=True)

# Sample checkpoints
out_dir = '../Data/checkpoint_results'
all_files = sorted(glob.glob(f'{out_dir}/*.csv'))

# Take every 10th file for speed
sampled_files = all_files[::10]
print(f"Sampling {len(sampled_files)} of {len(all_files)} files...")

llm_data_list = []
for i, f in enumerate(sampled_files):
    if i % 50 == 0:
        print(f"Progress: {i}/{len(sampled_files)}")
    df = pd.read_csv(f)
    llm_data_list.append(df)

llm_data = pd.concat(llm_data_list, ignore_index=True)

# Average across prompts
llm_data_avg = llm_data.groupby(['model', 'checkpoint', 'tokens']).agg({
    'preference': 'mean'
}).reset_index()

# Join with human data
merged = llm_data_avg.merge(avg_human_pref[['binom', 'hum_pref', 'Attested']], on='binom')

# Compute correlations
for item_type, attested_val in [('NONCE', 0), ('ATTESTED', 1)]:
    print(f"\n{'='*80}")
    print(f"{item_type} BINOMIALS")
    print('='*80)

    subset = merged[merged['Attested'] == attested_val]
    cors = subset.groupby(['model', 'tokens']).apply(
        lambda x: x['preference'].corr(x['hum_pref'])
    ).reset_index(name='correlation')

    for model in cors['model'].unique():
        model_data = cors[cors['model'] == model].sort_values('tokens')
        if len(model_data) > 0:
            print(f"\n{model}:")
            print(f"  Checkpoints sampled: {len(model_data)}")

            # Show first, mid, last
            if len(model_data) >= 3:
                first = model_data.iloc[0]
                mid = model_data.iloc[len(model_data)//2]
                last = model_data.iloc[-1]
                print(f"  Early ({first['tokens']/1e9:.1f}B tokens): r = {first['correlation']:.3f}")
                print(f"  Mid ({mid['tokens']/1e9:.1f}B tokens): r = {mid['correlation']:.3f}")
                print(f"  Final ({last['tokens']/1e9:.1f}B tokens): r = {last['correlation']:.3f}")
            elif len(model_data) > 0:
                last = model_data.iloc[-1]
                print(f"  Final ({last['tokens']/1e9:.1f}B tokens): r = {last['correlation']:.3f}")

print("\n\nDone!")
