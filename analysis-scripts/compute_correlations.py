#!/usr/bin/env python3
"""
Compute correlations between model and human preferences
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Load human data
human_data = pd.read_csv('../Data/all_human_data.csv')

# Create attested flag
human_data['Attested'] = (human_data['OverallFreq'] > 0).astype(int)

# Create response binary
human_data['resp_binary'] = human_data['resp'].map({'alpha': 1, 'nonalpha': 0})

# Create binomial column for joining
human_data['binom'] = human_data['Alpha']

# Average human preferences per binomial
avg_human_pref = human_data.groupby('binom').agg({
    'resp_binary': 'mean',
    'Attested': 'first',
    'Word1': 'first',
    'Word2': 'first'
}).reset_index()
avg_human_pref.rename(columns={'resp_binary': 'hum_pref'}, inplace=True)

# Load BabyLM unigrams for filtering
babylm_unigrams = pd.read_csv('../Data/babylm_eng_unigrams.csv')
babylm_unigrams['word'] = babylm_unigrams['word'].str.lower()

# Function to clean words
def clean_word(x):
    return str(x).lower().strip('.,!?;:"\'')

avg_human_pref['Word1_clean'] = avg_human_pref['Word1'].apply(clean_word)
avg_human_pref['Word2_clean'] = avg_human_pref['Word2'].apply(clean_word)

# Join with BabyLM unigrams
avg_human_pref = avg_human_pref.merge(
    babylm_unigrams.rename(columns={'count': 'Word1BabyLMCount'}),
    left_on='Word1_clean',
    right_on='word',
    how='left'
).drop('word', axis=1)

avg_human_pref = avg_human_pref.merge(
    babylm_unigrams.rename(columns={'count': 'Word2BabyLMCount'}),
    left_on='Word2_clean',
    right_on='word',
    how='left'
).drop('word', axis=1)

avg_human_pref['Word1BabyLMCount'] = avg_human_pref['Word1BabyLMCount'].fillna(0)
avg_human_pref['Word2BabyLMCount'] = avg_human_pref['Word2BabyLMCount'].fillna(0)

# Filter to binomials where both words appear in BabyLM
avg_human_pref = avg_human_pref[
    (avg_human_pref['Word1BabyLMCount'] > 0) &
    (avg_human_pref['Word2BabyLMCount'] > 0)
]

print(f"Human data loaded: {len(avg_human_pref)} binomials")
print(f"Nonce: {(avg_human_pref['Attested'] == 0).sum()}")
print(f"Attested: {(avg_human_pref['Attested'] == 1).sum()}")

# Load LLM data
out_dir = '../Data/checkpoint_results'
csv_files = glob.glob(f'{out_dir}/*.csv')

print(f"\nLoading {len(csv_files)} checkpoint files...")

llm_data_list = []
for i, f in enumerate(csv_files):
    if i % 500 == 0:
        print(f"  Progress: {i}/{len(csv_files)}")
    df = pd.read_csv(f)
    llm_data_list.append(df)

llm_data = pd.concat(llm_data_list, ignore_index=True)

print(f"LLM data loaded: {len(llm_data)} rows")

# Average preferences across prompts
llm_data_avg = llm_data.groupby(['model', 'binom', 'checkpoint', 'step', 'tokens']).agg({
    'preference': 'mean'
}).reset_index()

print(f"After averaging across prompts: {len(llm_data_avg)} rows")

# Join with human data
merged = llm_data_avg.merge(avg_human_pref[['binom', 'hum_pref', 'Attested']], on='binom', how='left')

# Remove rows without human preferences
merged = merged.dropna(subset=['hum_pref'])

print(f"After merging with human data: {len(merged)} rows")

# Compute correlations for each checkpoint
nonce_data = merged[merged['Attested'] == 0]
attested_data = merged[merged['Attested'] == 1]

nonce_cors = nonce_data.groupby(['model', 'checkpoint', 'tokens']).apply(
    lambda x: pd.Series({
        'hum_model_cor': x['preference'].corr(x['hum_pref']),
        'n_items': len(x)
    })
).reset_index()

attested_cors = attested_data.groupby(['model', 'checkpoint', 'tokens']).apply(
    lambda x: pd.Series({
        'hum_model_cor': x['preference'].corr(x['hum_pref']),
        'n_items': len(x)
    })
).reset_index()

print("\n" + "="*80)
print("NONCE BINOMIALS - Final Correlations by Model")
print("="*80)
for model in nonce_cors['model'].unique():
    model_data = nonce_cors[nonce_cors['model'] == model].sort_values('tokens')
    if len(model_data) > 0:
        final_cor = model_data.iloc[-1]['hum_model_cor']
        max_tokens = model_data.iloc[-1]['tokens']
        print(f"\n{model}:")
        print(f"  Final correlation: {final_cor:.3f}")
        print(f"  At tokens: {max_tokens:,}")
        print(f"  N checkpoints: {len(model_data)}")

        # Show trend (first, middle, last)
        if len(model_data) >= 3:
            first = model_data.iloc[0]
            mid = model_data.iloc[len(model_data)//2]
            last = model_data.iloc[-1]
            print(f"  Trend: {first['hum_model_cor']:.3f} -> {mid['hum_model_cor']:.3f} -> {last['hum_model_cor']:.3f}")

print("\n" + "="*80)
print("ATTESTED BINOMIALS - Final Correlations by Model")
print("="*80)
for model in attested_cors['model'].unique():
    model_data = attested_cors[attested_cors['model'] == model].sort_values('tokens')
    if len(model_data) > 0:
        final_cor = model_data.iloc[-1]['hum_model_cor']
        max_tokens = model_data.iloc[-1]['tokens']
        print(f"\n{model}:")
        print(f"  Final correlation: {final_cor:.3f}")
        print(f"  At tokens: {max_tokens:,}")
        print(f"  N checkpoints: {len(model_data)}")

        # Show trend
        if len(model_data) >= 3:
            first = model_data.iloc[0]
            mid = model_data.iloc[len(model_data)//2]
            last = model_data.iloc[-1]
            print(f"  Trend: {first['hum_model_cor']:.3f} -> {mid['hum_model_cor']:.3f} -> {last['hum_model_cor']:.3f}")

# Save summary
nonce_cors['item_type'] = 'nonce'
attested_cors['item_type'] = 'attested'
all_cors = pd.concat([nonce_cors, attested_cors])
all_cors.to_csv('correlation_summary.csv', index=False)
print(f"\n\nSaved correlation summary to correlation_summary.csv")
