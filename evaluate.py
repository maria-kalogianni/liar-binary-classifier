"""
evaluate.py — Batch evaluation of TruthClassifier on a CSV file.

Usage (from project root):
    python evaluate.py data.csv

What it does:
    1. Loads and deduplicates the CSV
    2. Maps 6-class labels to binary TRUE/FALSE
    3. Runs batch inference (NO explainability — fast)
    4. Prints all metrics: Accuracy, Balanced Accuracy, AUC, MCC, F1, Precision, Recall
    5. Prints confusion matrix

Why no explainability: predict() with LayerIG + GradientSHAP takes ~60-90 seconds
per statement. For batch evaluation we only need the model logits, not explanations.
"""

import sys
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast

# Add parent folder to path so we can import truthfulness_classifier
sys.path.insert(0, os.path.dirname(__file__))

from truthfulness_classifier.model import TruthClassifier
from truthfulness_classifier.preprocess import (
    deduplicate, LABEL_MAP, standardize_format,
    compute_cs_lookup, build_meta_array,
    norm_speaker, norm_context, norm_affil,
)


ARTIFACTS_DIR = 'distilbert_model'
BATCH_SIZE    = 32   # larger batch = faster evaluation


class _EvalDataset(Dataset):
    """Simple dataset for batch evaluation — no labels required."""

    def __init__(self, df, cs_array, meta_array, tokenizer, max_len):
        self.df        = df.reset_index(drop=True)
        self.cs        = cs_array    # (n, 1) numpy array
        self.meta      = meta_array  # (n, 4) numpy array
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row['statement']) if str(row['statement']) != 'nan' else ''
        enc  = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'cs_ratio'      : torch.tensor([self.cs[idx]], dtype=torch.float32),
            'metadata'      : torch.tensor(self.meta[idx], dtype=torch.float32),
        }


def evaluate(csv_path, artifacts_dir=ARTIFACTS_DIR):
    """
    Evaluate TruthClassifier on a CSV file and print metrics.

    Args:
        csv_path      : path to data.csv (must have 'statement' and 'label' columns)
        artifacts_dir : folder with best_model.pt, preprocessors.pkl, tokenizer/
    """
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, roc_auc_score,
        matthews_corrcoef, f1_score, precision_score, recall_score,
        confusion_matrix,
    )

    print(f'\n=== TruthClassifier Evaluation ===')
    print(f'CSV       : {csv_path}')
    print(f'Artifacts : {artifacts_dir}/')

    # ── 1. Load artifacts ─────────────────────────────────────────────────────
    pkl_path = os.path.join(artifacts_dir, 'preprocessors.pkl')
    with open(pkl_path, 'rb') as f:
        artifacts = pickle.load(f)

    cfg       = artifacts.get('model_config', {
        'model_name': 'distilbert-base-uncased',
        'hidden_dim': 256,
        'dropout'   : 0.27,
        'max_len'   : 128,
    })
    threshold    = artifacts.get('best_threshold', 0.57)
    default_cred = artifacts.get('default_cred', 0.557)
    cs_default   = artifacts.get('cs_default', 0.5)
    len_mean     = artifacts.get('len_mean', 50.0)
    len_std      = artifacts.get('len_std',  20.0)

    # Pre-normalise lookup keys (same logic as predict.py)
    speaker_cs_norm   = {norm_speaker(k): v for k, v in artifacts.get('speaker_cs_lookup',   {}).items()}
    speaker_cred_norm = {norm_speaker(k): v for k, v in artifacts.get('speaker_cred_lookup', {}).items()}
    affil_cred_norm   = {norm_affil(k):   v for k, v in artifacts.get('affil_cred_lookup',   {}).items()}
    ctx_cred_norm     = {norm_context(k): v for k, v in artifacts.get('ctx_cred_lookup',     {}).items()}

    # ── 2. Load CSV ───────────────────────────────────────────────────────────
    print('\nLoading CSV...')
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['label'] = df['label'].str.lower().str.strip()
    print(f'Loaded {len(df)} rows.')

    has_labels = 'label' in df.columns and df['label'].notna().any()

    # ── 3. Deduplication (same as training pipeline) ──────────────────────────
    print('Deduplicating...')
    df = deduplicate(df)

    # ── 4. Map labels to binary (if labels present) ───────────────────────────
    if has_labels:
        df['binary_label'] = df['label'].map(LABEL_MAP)
        df = df.dropna(subset=['binary_label']).copy()
        df['binary_label'] = df['binary_label'].astype(int)
        n_true  = int(df['binary_label'].sum())
        n_false = len(df) - n_true
        print(f'Binary labels: TRUE={n_true}, FALSE={n_false}')
    else:
        print('No labels found — running prediction only (no metrics).')

    df = df.reset_index(drop=True)

    # ── 5. Build features ─────────────────────────────────────────────────────
    print('Building features...')

    # Credit Score for each row (using training lookup, fallback=cs_default)
    def _get_cs(speaker_name):
        if pd.isna(speaker_name):
            return cs_default
        return speaker_cs_norm.get(norm_speaker(str(speaker_name)), cs_default)

    cs_array = df['speaker_name'].apply(_get_cs).values.astype(np.float32)  # (n,)

    # 4-column metadata array
    meta_array = build_meta_array(
        df,
        {norm_speaker(k): v for k, v in artifacts.get('speaker_cred_lookup', {}).items()},
        {norm_affil(k):   v for k, v in artifacts.get('affil_cred_lookup',   {}).items()},
        {norm_context(k): v for k, v in artifacts.get('ctx_cred_lookup',     {}).items()},
        len_mean, len_std, default_cred,
    )

    # ── 6. Load model ─────────────────────────────────────────────────────────
    print('Loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = TruthClassifier(cfg['model_name'], cfg['hidden_dim'], cfg['dropout']).to(device)
    model_path = os.path.join(artifacts_dir, 'best_model.pt')
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── 7. Load tokenizer ─────────────────────────────────────────────────────
    tok_path = os.path.join(artifacts_dir, 'tokenizer')
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(tok_path)
    except Exception:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # ── 8. Batch inference ────────────────────────────────────────────────────
    print(f'Running inference on {len(df)} rows (batch_size={BATCH_SIZE})...')

    dataset = _EvalDataset(df, cs_array, meta_array, tokenizer, cfg['max_len'])
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['cs_ratio'].to(device),
                batch['metadata'].to(device),
            )
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs)
            # Progress indicator every 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                print(f'  Batch {i+1}/{len(loader)}', end='\r')

    print()
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= threshold).astype(int)

    # ── 9. Print metrics ──────────────────────────────────────────────────────
    if has_labels:
        y_true = df['binary_label'].values

        acc    = accuracy_score(y_true, all_preds)
        bal    = balanced_accuracy_score(y_true, all_preds)
        auc    = roc_auc_score(y_true, all_probs)
        mcc    = matthews_corrcoef(y_true, all_preds)
        f1     = f1_score(y_true, all_preds)
        prec   = precision_score(y_true, all_preds)
        rec    = recall_score(y_true, all_preds)
        cm     = confusion_matrix(y_true, all_preds)

        print(f'\n=== RESULTS (threshold={threshold:.2f}) ===')
        print(f'  Accuracy          : {acc:.4f}')
        print(f'  Balanced Accuracy : {bal:.4f}')
        print(f'  AUC-ROC           : {auc:.4f}')
        print(f'  MCC               : {mcc:.4f}')
        print(f'  F1                : {f1:.4f}')
        print(f'  Precision         : {prec:.4f}')
        print(f'  Recall            : {rec:.4f}')
        print(f'\nConfusion Matrix (rows=actual, cols=predicted):')
        print(f'            Pred FALSE  Pred TRUE')
        print(f'  True FALSE    {cm[0,0]:>5}      {cm[0,1]:>5}')
        print(f'  True TRUE     {cm[1,0]:>5}      {cm[1,1]:>5}')
    else:
        # No labels — just print predictions
        print(f'\nPredictions (threshold={threshold:.2f}):')
        df['prediction'] = ['TRUE' if p == 1 else 'FALSE' for p in all_preds]
        df['probability'] = all_probs.round(4)
        print(df[['statement', 'prediction', 'probability']].head(20).to_string())
        print(f'\n... ({len(df)} total predictions)')

    print('\nDone.')
    return all_probs, all_preds


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python evaluate.py <path_to_data.csv>')
        print('Example: python evaluate.py data.csv')
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f'ERROR: File not found: {csv_path}')
        sys.exit(1)

    evaluate(csv_path)
