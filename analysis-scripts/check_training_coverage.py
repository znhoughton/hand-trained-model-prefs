#!/usr/bin/env python3
"""
Check which binomial words and binomials appear in training corpora.

For each binomial:
- Check if Word1 and Word2 appear in corpus unigrams
- Check if binomial appears in corpus (either "word1 and word2" or "word2 and word1")
- Collect frequency statistics

Output: CSV with coverage statistics per binomial
"""

import pandas as pd
import sys
from pathlib import Path

# UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def clean_word(word):
    """Clean word by lowercasing and stripping punctuation."""
    return str(word).lower().strip('.,!?;:"\\'')


def check_babylm_coverage(binomials_df, unigrams_path, trigrams_path):
    """
    Check BabyLM corpus coverage for binomials.

    Args:
        binomials_df: DataFrame with Word1, Word2, Alpha columns
        unigrams_path: Path to babylm_eng_unigrams.csv
        trigrams_path: Path to babylm_eng_trigrams.csv

    Returns:
        DataFrame with coverage statistics
    """
    print(f"\nLoading BabyLM unigrams from {unigrams_path}...")
    unigrams = pd.read_csv(unigrams_path)
    unigrams['word'] = unigrams['word'].str.lower()
    unigram_dict = dict(zip(unigrams['word'], unigrams['count']))

    print(f"Loaded {len(unigrams):,} unique unigrams")

    print(f"\nLoading BabyLM trigrams from {trigrams_path}...")
    trigrams = pd.read_csv(trigrams_path)
    # Lowercase trigram for matching
    trigrams['trigram_lower'] = trigrams['trigram'].str.lower()
    trigram_dict = dict(zip(trigrams['trigram_lower'], trigrams['count']))

    print(f"Loaded {len(trigrams):,} unique trigrams")

    results = []

    print(f"\nChecking coverage for {len(binomials_df)} binomials...")

    for idx, row in binomials_df.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(binomials_df)}")

        binom = row['Alpha']
        word1_raw = row['Word1']
        word2_raw = row['Word2']

        # Clean words
        word1 = clean_word(word1_raw)
        word2 = clean_word(word2_raw)

        # Check unigram coverage
        word1_freq = unigram_dict.get(word1, 0)
        word2_freq = unigram_dict.get(word2, 0)
        word1_in_corpus = word1_freq > 0
        word2_in_corpus = word2_freq > 0

        # Check trigram coverage (either ordering)
        # Try: "word1 and word2" and "word2 and word1"
        trigram_alpha = f"{word1} and {word2}"
        trigram_nonalpha = f"{word2} and {word1}"

        freq_alpha = trigram_dict.get(trigram_alpha, 0)
        freq_nonalpha = trigram_dict.get(trigram_nonalpha, 0)

        binomial_in_corpus = (freq_alpha > 0) or (freq_nonalpha > 0)
        binomial_freq = freq_alpha + freq_nonalpha

        # Which ordering was found?
        if freq_alpha > 0 and freq_nonalpha > 0:
            ordering_found = "both"
        elif freq_alpha > 0:
            ordering_found = "alphabetical"
        elif freq_nonalpha > 0:
            ordering_found = "non-alphabetical"
        else:
            ordering_found = "neither"

        results.append({
            'binom': binom,
            'Word1': word1_raw,
            'Word2': word2_raw,
            'Word1_clean': word1,
            'Word2_clean': word2,
            'Word1_in_corpus': word1_in_corpus,
            'Word2_in_corpus': word2_in_corpus,
            'Word1_freq': word1_freq,
            'Word2_freq': word2_freq,
            'both_words_in_corpus': word1_in_corpus and word2_in_corpus,
            'binomial_in_corpus': binomial_in_corpus,
            'binomial_freq': binomial_freq,
            'alpha_freq': freq_alpha,
            'nonalpha_freq': freq_nonalpha,
            'ordering_found': ordering_found,
            'corpus': 'BabyLM'
        })

    return pd.DataFrame(results)


def check_c4_coverage(binomials_df, unigrams_path=None, trigrams_path=None):
    """
    Check C4 corpus coverage for binomials.

    NOTE: If C4 unigrams/trigrams are not available, returns a placeholder
    indicating data is needed.

    Args:
        binomials_df: DataFrame with Word1, Word2, Alpha columns
        unigrams_path: Path to c4_unigrams.csv (if available)
        trigrams_path: Path to c4_trigrams.csv (if available)

    Returns:
        DataFrame with coverage statistics (or placeholder if data unavailable)
    """
    if unigrams_path is None or trigrams_path is None:
        print("\nWARNING: C4 corpus data not provided.")
        print("Returning placeholder results. Please provide:")
        print("  - C4 unigrams CSV (word, count)")
        print("  - C4 trigrams CSV (trigram, count)")

        # Return placeholder
        results = []
        for idx, row in binomials_df.iterrows():
            results.append({
                'binom': row['Alpha'],
                'Word1': row['Word1'],
                'Word2': row['Word2'],
                'Word1_clean': clean_word(row['Word1']),
                'Word2_clean': clean_word(row['Word2']),
                'Word1_in_corpus': None,
                'Word2_in_corpus': None,
                'Word1_freq': None,
                'Word2_freq': None,
                'both_words_in_corpus': None,
                'binomial_in_corpus': None,
                'binomial_freq': None,
                'alpha_freq': None,
                'nonalpha_freq': None,
                'ordering_found': 'data_unavailable',
                'corpus': 'C4'
            })
        return pd.DataFrame(results)

    # If paths provided, use same logic as BabyLM
    print(f"\nLoading C4 unigrams from {unigrams_path}...")
    unigrams = pd.read_csv(unigrams_path)
    unigrams['word'] = unigrams['word'].str.lower()
    unigram_dict = dict(zip(unigrams['word'], unigrams['count']))

    print(f"Loaded {len(unigrams):,} unique unigrams")

    print(f"\nLoading C4 trigrams from {trigrams_path}...")
    trigrams = pd.read_csv(trigrams_path)
    trigrams['trigram_lower'] = trigrams['trigram'].str.lower()
    trigram_dict = dict(zip(trigrams['trigram_lower'], trigrams['count']))

    print(f"Loaded {len(trigrams):,} unique trigrams")

    results = []

    print(f"\nChecking coverage for {len(binomials_df)} binomials...")

    for idx, row in binomials_df.iterrows():
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(binomials_df)}")

        binom = row['Alpha']
        word1_raw = row['Word1']
        word2_raw = row['Word2']

        word1 = clean_word(word1_raw)
        word2 = clean_word(word2_raw)

        word1_freq = unigram_dict.get(word1, 0)
        word2_freq = unigram_dict.get(word2, 0)
        word1_in_corpus = word1_freq > 0
        word2_in_corpus = word2_freq > 0

        trigram_alpha = f"{word1} and {word2}"
        trigram_nonalpha = f"{word2} and {word1}"

        freq_alpha = trigram_dict.get(trigram_alpha, 0)
        freq_nonalpha = trigram_dict.get(trigram_nonalpha, 0)

        binomial_in_corpus = (freq_alpha > 0) or (freq_nonalpha > 0)
        binomial_freq = freq_alpha + freq_nonalpha

        if freq_alpha > 0 and freq_nonalpha > 0:
            ordering_found = "both"
        elif freq_alpha > 0:
            ordering_found = "alphabetical"
        elif freq_nonalpha > 0:
            ordering_found = "non-alphabetical"
        else:
            ordering_found = "neither"

        results.append({
            'binom': binom,
            'Word1': word1_raw,
            'Word2': word2_raw,
            'Word1_clean': word1,
            'Word2_clean': word2,
            'Word1_in_corpus': word1_in_corpus,
            'Word2_in_corpus': word2_in_corpus,
            'Word1_freq': word1_freq,
            'Word2_freq': word2_freq,
            'both_words_in_corpus': word1_in_corpus and word2_in_corpus,
            'binomial_in_corpus': binomial_in_corpus,
            'binomial_freq': binomial_freq,
            'alpha_freq': freq_alpha,
            'nonalpha_freq': freq_nonalpha,
            'ordering_found': ordering_found,
            'corpus': 'C4'
        })

    return pd.DataFrame(results)


def main():
    print("="*80)
    print("CHECKING TRAINING CORPUS COVERAGE FOR BINOMIALS")
    print("="*80)

    # Load binomial data
    print("\nLoading binomial data from ../Data/all_human_data.csv...")
    human_data = pd.read_csv('../Data/all_human_data.csv')

    # Get unique binomials
    binomials = human_data[['Alpha', 'Word1', 'Word2', 'OverallFreq']].drop_duplicates()
    binomials['Attested'] = binomials['OverallFreq'] > 0

    print(f"\nFound {len(binomials)} unique binomials:")
    print(f"  - Nonce: {(binomials['Attested'] == False).sum()}")
    print(f"  - Attested: {(binomials['Attested'] == True).sum()}")

    # Check BabyLM coverage
    print("\n" + "="*80)
    print("BABYLM CORPUS COVERAGE")
    print("="*80)

    babylm_coverage = check_babylm_coverage(
        binomials,
        unigrams_path='../Data/babylm_eng_unigrams.csv',
        trigrams_path='../Data/babylm_eng_trigrams.csv'
    )

    # Save BabyLM results
    babylm_output = '../Data/babylm_binomial_coverage.csv'
    babylm_coverage.to_csv(babylm_output, index=False)
    print(f"\nSaved BabyLM coverage to {babylm_output}")

    # Print BabyLM summary
    print("\n" + "-"*80)
    print("BABYLM SUMMARY")
    print("-"*80)
    print(f"Words in corpus:")
    print(f"  - Word1 found: {babylm_coverage['Word1_in_corpus'].sum()} / {len(babylm_coverage)} ({100*babylm_coverage['Word1_in_corpus'].mean():.1f}%)")
    print(f"  - Word2 found: {babylm_coverage['Word2_in_corpus'].sum()} / {len(babylm_coverage)} ({100*babylm_coverage['Word2_in_corpus'].mean():.1f}%)")
    print(f"  - Both words found: {babylm_coverage['both_words_in_corpus'].sum()} / {len(babylm_coverage)} ({100*babylm_coverage['both_words_in_corpus'].mean():.1f}%)")

    print(f"\nBinomials in corpus:")
    print(f"  - Found (any ordering): {babylm_coverage['binomial_in_corpus'].sum()} / {len(babylm_coverage)} ({100*babylm_coverage['binomial_in_corpus'].mean():.1f}%)")
    print(f"\nOrdering breakdown:")
    print(babylm_coverage['ordering_found'].value_counts())

    # Check C4 coverage (placeholder - update paths if C4 data available)
    print("\n" + "="*80)
    print("C4 CORPUS COVERAGE")
    print("="*80)

    # NOTE: Update these paths if C4 unigrams/trigrams are available
    c4_coverage = check_c4_coverage(
        binomials,
        unigrams_path=None,  # Update to '../Data/c4_unigrams.csv' if available
        trigrams_path=None   # Update to '../Data/c4_trigrams.csv' if available
    )

    # Save C4 results
    c4_output = '../Data/c4_binomial_coverage.csv'
    c4_coverage.to_csv(c4_output, index=False)
    print(f"\nSaved C4 coverage to {c4_output}")

    if c4_coverage['Word1_in_corpus'].isna().all():
        print("\nC4 data not available - placeholder file created.")
    else:
        # Print C4 summary
        print("\n" + "-"*80)
        print("C4 SUMMARY")
        print("-"*80)
        print(f"Words in corpus:")
        print(f"  - Word1 found: {c4_coverage['Word1_in_corpus'].sum()} / {len(c4_coverage)} ({100*c4_coverage['Word1_in_corpus'].mean():.1f}%)")
        print(f"  - Word2 found: {c4_coverage['Word2_in_corpus'].sum()} / {len(c4_coverage)} ({100*c4_coverage['Word2_in_corpus'].mean():.1f}%)")
        print(f"  - Both words found: {c4_coverage['both_words_in_corpus'].sum()} / {len(c4_coverage)} ({100*c4_coverage['both_words_in_corpus'].mean():.1f}%)")

        print(f"\nBinomials in corpus:")
        print(f"  - Found (any ordering): {c4_coverage['binomial_in_corpus'].sum()} / {len(c4_coverage)} ({100*c4_coverage['binomial_in_corpus'].mean():.1f}%)")
        print(f"\nOrdering breakdown:")
        print(c4_coverage['ordering_found'].value_counts())

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
