"""
preprocess.py — Data cleaning, deduplication, and feature engineering.

This module is used by both train.py (to prepare training data) and predict.py
(to normalise the user's input before feeding it to the model).

No model-specific logic here — just data transformations.
"""

import re
import pandas as pd
import numpy as np


# ── Label mapping ─────────────────────────────────────────────────────────────
# The dataset uses 6 fine-grained labels. We collapse them into binary TRUE/FALSE.
# Decision rationale :
#   half-true → TRUE because it is semantically
#   closer to truth. 
LABEL_MAP = {
    'true'            : 1,
    'mostly-true'     : 1,
    'half-true'       : 1,   
    'barely-true'     : 0,
    'false'           : 0,
    'extremely-false' : 0,
}

# Credit Score weights 
# Maps each label to a "falsehood score": 0.0 = fully truthful, 1.0 = fully false.
# Used to compute a per-speaker CS_ratio = weighted_sum / n_statements.
CS_WEIGHTS = {
    'true'            : 0.00,
    'mostly-true'     : 0.20,
    'half-true'       : 0.50,
    'barely-true'     : 0.75,
    'false'           : 0.90,
    'extremely-false' : 1.00,
}


# ── Text cleaning ─────────────────────────────────────────────────────────────

def standardize_format(text):
    """
    Light text cleaning for DistilBERT input.

    What we do and why:
    - Remove non-ASCII characters (smart quotes, em-dashes) → avoid tokenizer surprises
    - Remove URLs → not informative for truthfulness
    - Collapse whitespace → consistent input format

    We do NOT lemmatize or remove stop words — DistilBERT's WordPiece tokenizer
    handles morphology internally, and stop words carry syntactic context.

    Args:
        text : any value (will be converted to str if needed)

    Returns:
        cleaned string
    """
    # Handle NaN or None gracefully
    if pd.isna(text) or text is None:
        return ''
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # remove non-ASCII (e.g. smart quotes)
    text = re.sub(r'http\S+', '', text)            # remove URLs (http:// or https://)
    text = re.sub(r'\s+', ' ', text).strip()       # collapse multiple spaces/newlines to one
    return text


# ── Per-field normalisation functions ─────────────────────────────────────────
#
# CRITICAL: each CSV field is stored in a different format and needs its OWN
# normalisation function. Using one function for all would cause lookup failures.
#
#   speaker_name       stored as: 'barack-obama'   (hyphens, lowercase)
#   statement_context  stored as: 'a chain email'  (spaces + article, lowercase)
#   speaker_affiliation stored as: 'democrat'      (lowercase, no hyphens)
#
# Example of what goes wrong without field-specific norm:
#   'a chain email' normalised with norm_speaker → 'a-chain-email' → NOT FOUND in pkl
#   The correct key is 'a chain email' (spaces preserved) → norm_context finds it.

def norm_speaker(s):
    """
    Normalise a speaker name to match the format stored in preprocessors.pkl.
    Spaces are converted to hyphens because the CSV stores names as 'barack-obama'.

    Examples:
        'Barack Obama'   → 'barack-obama'
        'barack obama'   → 'barack-obama'
        'barack-obama'   → 'barack-obama'  (already correct, idempotent)
        ''               → ''
    """
    if not s or str(s).strip() == '':
        return ''
    # re.sub replaces one or more whitespace chars (\s+) with a single hyphen
    return re.sub(r'\s+', '-', str(s).lower().strip())


def norm_context(s):
    """
    Normalise a context string: lowercase + strip only.
    Do NOT convert spaces to hyphens — context keys have spaces in the pkl.

    Examples:
        'A Chain Email'    → 'a chain email'
        'a chain email'    → 'a chain email'  (idempotent)
        'Press Release'    → 'press release'
    """
    if not s or str(s).strip() == '':
        return ''
    return str(s).strip().lower()


def norm_affil(s):
    """
    Normalise a political affiliation string: lowercase + strip only.

    Examples:
        'Democrat'   → 'democrat'
        'Republican' → 'republican'
    """
    if not s or str(s).strip() == '':
        return ''
    return str(s).strip().lower()


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(df):
    """
    Remove duplicate statements BEFORE the train/test split.

    Why before the split: if we deduplicate after, the same statement text
    could appear in both train and test with different labels — data leakage.

    Two types of duplicates:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Type A: same (statement, label) → keep one copy (exact duplicate)  │
    │ Type B: same statement, different labels → drop ALL                │
    │         (contradictory annotations — untrustworthy for training)   │
    └─────────────────────────────────────────────────────────────────────┘

    ORDER MATTERS: must do Type A FIRST, then Type B.
    Reason: after Type A, each remaining (statement, label) pair is unique.
    Then Type B correctly identifies statements with genuinely conflicting labels.

    Args:
        df : pandas DataFrame with columns ['statement', 'label']

    Returns:
        cleaned DataFrame, reset index
    """
    n_before = len(df)

    # Type A: drop rows where the combination of (statement, label) is duplicated
    # keep='first' → keep the first occurrence, drop subsequent duplicates
    df = df.drop_duplicates(subset=['statement', 'label'], keep='first')
    n_after_a = len(df)

    # Type B: find statements that still appear with more than one unique label
    # groupby: groups all rows with the same statement text
    # nunique: counts how many distinct labels each statement has
    # loc[lambda x: x > 1]: keep only those with more than 1 unique label
    # .index: returns just the statement strings (the group keys)
    contradictory = (
        df.groupby('statement')['label']
          .nunique()
          .loc[lambda x: x > 1]
          .index
    )
    # ~df['statement'].isin(...) → keep rows where statement is NOT contradictory
    df = df[~df['statement'].isin(contradictory)].reset_index(drop=True)
    n_after_b = len(df)

    print(f'Deduplication: {n_before} -> {n_after_a} (Type A: -{n_before - n_after_a}) '
          f'-> {n_after_b} (Type B: -{n_after_a - n_after_b})')
    return df


# ── Feature computation (train set only) ─────────────────────────────────────

def compute_cs_lookup(train_df):
    """
    Compute Credit Score (CS) ratio for each speaker from the training set.

    Formula (Bhatt et al. 2021, Equation 1):
        CS_ratio = sum(cs_weights[label_i] for all statements by speaker) / n_statements

    Result interpretation:
        CS ≈ 0.0 → speaker is mostly truthful
        CS ≈ 1.0 → speaker is mostly false

    IMPORTANT: computed from training data ONLY to prevent data leakage into val/test.

    Args:
        train_df : training split DataFrame, must have columns ['speaker_name', 'label']

    Returns:
        dict: {speaker_name_string: cs_ratio_float}
        Keys are raw strings (not normalised) — normalisation happens at lookup time.
    """
    lookup = {}
    for speaker, group in train_df.groupby('speaker_name'):
        # Sum the falsehood weight for each statement by this speaker
        weighted_sum = sum(CS_WEIGHTS.get(lbl, 0.5) for lbl in group['label'])
        # Divide by total statements to get a ratio in [0, 1]
        lookup[str(speaker).strip()] = weighted_sum / len(group)
    return lookup


def compute_cred_lookups(train_df):
    """
    Compute TRUE rate (credibility score) for each unique speaker, affiliation,
    and context observed in the training set.

    Credibility score = fraction of TRUE statements for a given group.
    Example: if 'democrat' speakers are TRUE 62% of the time, affil_cred['democrat'] = 0.62

    Also computes statement length statistics (mean, std) for z-score normalisation.

    IMPORTANT: computed from training data ONLY.

    Args:
        train_df : training split DataFrame with columns:
                   ['speaker_name', 'speaker_affiliation', 'statement_context',
                    'statement', 'binary_label']

    Returns:
        dict with keys:
            'speaker_cred_lookup' : {speaker_name: TRUE_rate}
            'affil_cred_lookup'   : {affiliation: TRUE_rate}
            'ctx_cred_lookup'     : {context: TRUE_rate}
            'len_mean'            : float — mean statement length in characters
            'len_std'             : float — std of statement length (+ small epsilon)
            'default_cred'        : float — overall TRUE rate in train (fallback value)
    """
    # Overall TRUE rate in training set — used as fallback for unseen values
    default_cred = float(train_df['binary_label'].mean())

    def _build_cred_lookup(df, col):
        """
        For each unique value in column `col`, compute fraction of TRUE labels.
        Returns dict: {value_string: mean_binary_label}
        """
        return {
            str(val).strip(): float(grp['binary_label'].mean())
            for val, grp in df.groupby(col)
        }

    speaker_cred_lookup = _build_cred_lookup(train_df, 'speaker_name')
    affil_cred_lookup   = _build_cred_lookup(train_df, 'speaker_affiliation')
    ctx_cred_lookup     = _build_cred_lookup(train_df, 'statement_context')

    # Statement length statistics for z-score normalisation
    # fillna('') ensures NaN statements are treated as empty string (length 0)
    train_lens = train_df['statement'].fillna('').apply(len)
    len_mean   = float(train_lens.mean())
    # + 1e-8 prevents division by zero if all statements have identical length (unlikely)
    len_std    = float(train_lens.std()) + 1e-8

    return {
        'speaker_cred_lookup': speaker_cred_lookup,
        'affil_cred_lookup'  : affil_cred_lookup,
        'ctx_cred_lookup'    : ctx_cred_lookup,
        'len_mean'           : len_mean,
        'len_std'            : len_std,
        'default_cred'       : default_cred,
    }


def build_meta_array(df_split, speaker_cred_lookup, affil_cred_lookup,
                     ctx_cred_lookup, len_mean, len_std, default_cred):
    """
    Build the 4-column metadata feature array for a DataFrame split.

    The 4 features (in order):
        col 0 — speaker_cred : TRUE rate of this speaker (from training set)
        col 1 — affil_cred   : TRUE rate of this speaker's affiliation
        col 2 — ctx_cred     : TRUE rate of statements made in this context
        col 3 — len_norm     : z-score of statement character length

    For unseen values (not in the lookup), we use default_cred as fallback.

    Args:
        df_split             : any DataFrame split (train / val / test)
        speaker_cred_lookup  : dict from compute_cred_lookups
        affil_cred_lookup    : dict from compute_cred_lookups
        ctx_cred_lookup      : dict from compute_cred_lookups
        len_mean             : from compute_cred_lookups
        len_std              : from compute_cred_lookups
        default_cred         : fallback credibility value for unseen entries

    Returns:
        numpy array of shape (len(df_split), 4), dtype float32
    """
    d    = df_split.reset_index(drop=True)
    meta = np.zeros((len(d), 4), dtype=np.float32)

    def _lookup(val, lookup_dict):
        """Return the credibility for this value, or default_cred if not found."""
        if pd.isna(val):
            return default_cred
        return lookup_dict.get(str(val).strip(), default_cred)

    meta[:, 0] = d['speaker_name'].apply(lambda v: _lookup(v, speaker_cred_lookup)).values
    meta[:, 1] = d['speaker_affiliation'].apply(lambda v: _lookup(v, affil_cred_lookup)).values
    meta[:, 2] = d['statement_context'].apply(lambda v: _lookup(v, ctx_cred_lookup)).values

    # Z-score: (value - mean) / std  →  centred around 0, unit std
    lengths    = d['statement'].fillna('').apply(len).values
    meta[:, 3] = (lengths - len_mean) / len_std

    return meta
