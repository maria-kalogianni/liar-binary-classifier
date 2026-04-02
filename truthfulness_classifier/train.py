"""
train.py — Training pipeline for TruthClassifier.

Entry point: train(csv_path, artifacts_dir)

What it does:
    1. Load data.csv
    2. Deduplicate (must be before split to prevent leakage)
    3. Map 6-class labels to binary (TRUE=1, FALSE=0)
    4. Stratified 80 / 10 / 10 split
    5. Compute Credit Score + credibility lookups from training set ONLY
    6. Train DistilBERT + CS + metadata model for 2 epochs
    7. Save best weights (by validation Balanced Accuracy)
    8. Save tokenizer and preprocessors.pkl


"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .preprocess import (
    deduplicate,
    LABEL_MAP,
    CS_WEIGHTS,
    compute_cs_lookup,
    compute_cred_lookups,
    build_meta_array,
)
from .model import TruthClassifier

warnings.filterwarnings('ignore')

# Fixed random seed — ensures reproducible splits and weight initialisation
SEED = 42

# Optuna best hyperparameters (Day 5B/6 experiments on Kaggle)
# Do NOT change these unless you re-run Optuna
_LR           = 3.631079257973987e-05
_DROPOUT      = 0.26978082128377195
_HIDDEN_DIM   = 256
_WARMUP_RATIO = 0.08744379962617845
_N_EPOCHS     = 2          # best epoch is consistently 2  (see CLAUDE.md)
_MAX_LEN      = 128        # sufficient for short political statements
_BATCH_SIZE   = 16
_WEIGHT_DECAY = 0.01
_MODEL_NAME   = 'distilbert-base-uncased'
_CS_DEFAULT   = 0.5        # fallback CS for unseen speakers — hardcoded, not in pkl


class _TruthDataset(Dataset):
    """
    PyTorch Dataset that feeds batches to the training loop.

    Each item contains:
        input_ids      : (max_len,)  integer token IDs
        attention_mask : (max_len,)  1 for real tokens, 0 for [PAD]
        cs_ratio       : (1,)        credit score for the speaker
        metadata       : (4,)        [speaker_cred, affil_cred, ctx_cred, len_norm]
        label          : scalar float 0.0 or 1.0
    """

    def __init__(self, df, meta_array, tokenizer, max_len):
        """
        Args:
            df         : DataFrame split (train / val / test)
            meta_array : numpy array (n_rows, 4) from build_meta_array()
            tokenizer  : DistilBertTokenizer
            max_len    : maximum token sequence length
        """
        self.df        = df.reset_index(drop=True)
        self.meta      = meta_array   # numpy (n, 4) — one row per sample
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row['statement']) if pd.notna(row['statement']) else ''

        # Tokenise: convert text → token IDs + attention mask
        # padding='max_length' → pad short sequences with [PAD] (token ID 0)
        # truncation=True → cut sequences longer than max_len
        # return_tensors='pt' → return PyTorch tensors, not lists
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            # squeeze(0): tokenizer adds a batch dimension → (1, max_len) → (max_len,)
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'cs_ratio'      : torch.tensor([row['cs_ratio']], dtype=torch.float32),
            'metadata'      : torch.tensor(self.meta[idx], dtype=torch.float32),
            'label'         : torch.tensor(row['binary_label'], dtype=torch.float32),
        }


def _train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Run one full training epoch over the DataLoader.
    Returns the average loss across all batches.
    """
    model.train()   # training mode: Dropout is ACTIVE
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cs_ratio       = batch['cs_ratio'].to(device)
        metadata       = batch['metadata'].to(device)
        # unsqueeze(1): label is (batch,) but loss expects (batch, 1) to match logits
        labels         = batch['label'].unsqueeze(1).to(device)

        optimizer.zero_grad()   # clear gradients from the previous batch

        logits = model(input_ids, attention_mask, cs_ratio, metadata)
        loss   = criterion(logits, labels)

        loss.backward()   # compute gradients via backpropagation

        # Gradient clipping: prevents "exploding gradients" common in transformers.
        # If the gradient norm > 1.0, scale all gradients down proportionally.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()    # update model weights
        scheduler.step()    # update learning rate (linear warmup + decay)

        total_loss += loss.item()   # .item() converts tensor to Python float

    return total_loss / len(loader)   # average loss per batch


def _run_inference(model, loader, device):
    """
    Run the model on all batches in `loader` without computing gradients.
    Returns:
        y_true  : numpy array of true binary labels (0 or 1)
        y_probs : numpy array of P(TRUE) probabilities in [0, 1]
    """
    model.eval()   # eval mode: Dropout is OFF → deterministic outputs
    all_labels = []
    all_probs  = []

    with torch.no_grad():   # disables gradient tracking — saves memory + speeds up
        for batch in loader:
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['cs_ratio'].to(device),
                batch['metadata'].to(device),
            )
            # sigmoid: converts raw logit to probability in [0, 1]
            # squeeze(1): (batch, 1) → (batch,)
            # .cpu().numpy(): move from GPU to CPU and convert to numpy
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_labels.extend(batch['label'].numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def train(csv_path, artifacts_dir='distilbert_model'):
    """
    Train TruthClassifier on the   dataset and save all artifacts.

    Args:
        csv_path      : path to data.csv (e.g. 'data.csv' or full path)
        artifacts_dir : folder where all outputs will be saved

    Outputs (inside artifacts_dir/):
        best_model.pt       — model weights (best validation balanced accuracy)
        tokenizer/          — DistilBERT tokenizer files
        preprocessors.pkl   — credibility lookups, length stats, threshold, config

    Expected console output when successful:
        Device: cuda  (or cpu)
        Loaded 12836 rows, 8 columns.
        Deduplication: ... → ... → ...
        Binary labels: TRUE=5367, FALSE=4260, TRUE rate=0.557
        Split: train=7700 | val=963 | test=964
        Epoch 1/2 | Loss=0.XXXX | Val Bal.Acc=0.XXXX | AUC=0.XXXX
        Epoch 2/2 | Loss=0.XXXX | Val Bal.Acc=0.XXXX | AUC=0.XXXX
        Best threshold (by val balanced accuracy): 0.55
        All artifacts saved to: distilbert_model/
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f'\nLoading {csv_path} ...')
    df = pd.read_csv(csv_path)

    # Normalise column names: remove spaces and make lowercase
    # (robustness against CSV files with slightly different formatting)
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalise label values: lowercase + strip whitespace
    df['label'] = df['label'].str.lower().str.strip()

    print(f'Loaded {len(df)} rows, {len(df.columns)} columns.')
    print('Label distribution:')
    print(df['label'].value_counts().to_string())

    # ── 2. Deduplication (BEFORE split — critical to prevent leakage) ─────────
    df = deduplicate(df)

    # ── 3. Binary label mapping ───────────────────────────────────────────────
    df['binary_label'] = df['label'].map(LABEL_MAP)
    # Drop rows with labels not in LABEL_MAP (unknown labels — shouldn't happen)
    n_before = len(df)
    df = df.dropna(subset=['binary_label']).copy()
    if len(df) < n_before:
        print(f'Dropped {n_before - len(df)} rows with unknown labels.')
    df['binary_label'] = df['binary_label'].astype(int)
    df = df.reset_index(drop=True)

    n_true  = int(df['binary_label'].sum())
    n_false = len(df) - n_true
    print(f'\nBinary labels: TRUE={n_true}, FALSE={n_false}, TRUE rate={n_true/len(df):.3f}')

    # ── 4. Stratified 80 / 10 / 10 split ─────────────────────────────────────
    # stratify= ensures each split has the same TRUE/FALSE ratio
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.10,
        random_state=SEED,
        stratify=df['binary_label'],
    )
    # val is 10% of total = 10/90 ≈ 0.111 of train_val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.10 / 0.90,
        random_state=SEED,
        stratify=train_val_df['binary_label'],
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f'Split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}')
    print(f'TRUE rates: train={train_df["binary_label"].mean():.3f} | '
          f'val={val_df["binary_label"].mean():.3f} | '
          f'test={test_df["binary_label"].mean():.3f}')

    # ── 5. Feature computation from training set ONLY ─────────────────────────
    print('\nComputing Credit Score and credibility lookups from training set...')

    # Credit Score lookup: {speaker_name_raw_string: cs_ratio_float}
    speaker_cs_lookup = compute_cs_lookup(train_df)

    # Add cs_ratio column to all splits (using training lookup, fallback=0.5)
    def _get_cs(speaker_name):
        if pd.isna(speaker_name):
            return _CS_DEFAULT
        return speaker_cs_lookup.get(str(speaker_name).strip(), _CS_DEFAULT)

    train_df['cs_ratio'] = train_df['speaker_name'].apply(_get_cs)
    val_df['cs_ratio']   = val_df['speaker_name'].apply(_get_cs)
    test_df['cs_ratio']  = test_df['speaker_name'].apply(_get_cs)

    # Credibility lookups and length statistics (all from train only)
    cred_data = compute_cred_lookups(train_df)

    print(f'Credibility default (overall train TRUE rate): {cred_data["default_cred"]:.3f}')
    print(f'Unique speakers with CS: {len(speaker_cs_lookup)}')

    # Build 4-column metadata arrays for each split
    meta_train = build_meta_array(
        train_df,
        cred_data['speaker_cred_lookup'], cred_data['affil_cred_lookup'],
        cred_data['ctx_cred_lookup'], cred_data['len_mean'], cred_data['len_std'],
        cred_data['default_cred'],
    )
    meta_val = build_meta_array(
        val_df,
        cred_data['speaker_cred_lookup'], cred_data['affil_cred_lookup'],
        cred_data['ctx_cred_lookup'], cred_data['len_mean'], cred_data['len_std'],
        cred_data['default_cred'],
    )

    # ── 6. Tokeniser and DataLoaders ──────────────────────────────────────────
    print(f'\nLoading tokeniser: {_MODEL_NAME} ...')
    tokenizer = DistilBertTokenizer.from_pretrained(_MODEL_NAME)

    train_dataset = _TruthDataset(train_df, meta_train, tokenizer, _MAX_LEN)
    val_dataset   = _TruthDataset(val_df,   meta_val,   tokenizer, _MAX_LEN)

    # num_workers=0: safest on Windows (avoids multiprocessing spawn issues).
    # On Linux/Kaggle you can increase to 2-4 for faster data loading.
    train_loader = DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=_BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}')

    # ── 7. Model, loss function, optimiser, scheduler ─────────────────────────
    print(f'\nBuilding TruthClassifier (hidden_dim={_HIDDEN_DIM}, dropout={_DROPOUT:.3f})...')
    model = TruthClassifier(_MODEL_NAME, _HIDDEN_DIM, _DROPOUT).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')

    # pos_weight compensates for class imbalance.
    # BCEWithLogitsLoss penalises FALSE negatives more when TRUE is rarer.
    # Formula: n_false / n_true — if FALSE > TRUE, TRUE gets more weight.
    n_true_tr  = int(train_df['binary_label'].sum())
    n_false_tr = len(train_df) - n_true_tr
    pos_weight = torch.tensor([n_false_tr / n_true_tr], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f'pos_weight = {pos_weight.item():.3f} (n_false/n_true)')

    # AdamW = Adam with weight decay — standard choice for transformer fine-tuning
    optimizer = AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)

    total_steps  = len(train_loader) * _N_EPOCHS
    warmup_steps = int(total_steps * _WARMUP_RATIO)
    # Linear warmup: avoids unstable large updates at the start of training.
    # After warmup, learning rate decays linearly to 0 by the end of training.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f'Total steps: {total_steps}, warmup steps: {warmup_steps}')

    # ── 8. Training loop ──────────────────────────────────────────────────────
    print(f'\nTraining for {_N_EPOCHS} epochs...')

    best_val_ba  = 0.0      # best validation balanced accuracy seen so far
    best_weights = None     # copy of best model weights (in memory)
    best_epoch   = 0

    for epoch in range(1, _N_EPOCHS + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_labels, val_probs = _run_inference(model, val_loader, device)

        # Balanced accuracy = average of sensitivity and specificity
        # Better than plain accuracy for imbalanced classes
        val_ba  = balanced_accuracy_score(val_labels, (val_probs >= 0.5).astype(int))
        val_auc = roc_auc_score(val_labels, val_probs)

        print(f'Epoch {epoch}/{_N_EPOCHS} | '
              f'Loss={train_loss:.4f} | '
              f'Val Bal.Acc={val_ba:.4f} | '
              f'Val AUC={val_auc:.4f}')

        # Save best model by validation balanced accuracy
        if val_ba > best_val_ba:
            best_val_ba  = val_ba
            best_epoch   = epoch
            # .cpu().clone(): move weights to CPU memory and make a copy
            # (model continues training on GPU — this snapshot is independent)
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  >> Best model saved (val Bal.Acc = {best_val_ba:.4f})')

    print(f'\nTraining complete. Best epoch: {best_epoch}')

    # ── 9. Threshold tuning on validation set ─────────────────────────────────
    # Load best weights back and find the classification threshold that maximises
    # balanced accuracy on the validation set.
    model.load_state_dict(best_weights)
    val_labels, val_probs = _run_inference(model, val_loader, device)

    thresholds = np.arange(0.30, 0.71, 0.01)   # test thresholds from 0.30 to 0.70
    bal_accs   = [
        balanced_accuracy_score(val_labels, (val_probs >= t).astype(int))
        for t in thresholds
    ]
    # np.argmax returns the index of the maximum value
    best_t = float(thresholds[int(np.argmax(bal_accs))])
    print(f'Best threshold (by val balanced accuracy): {best_t:.2f}')

    # ── 10. Save artifacts ────────────────────────────────────────────────────
    print(f'\nSaving artifacts to {artifacts_dir}/ ...')
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save model weights (state_dict = all learned parameters as a dict of tensors)
    model_path = os.path.join(artifacts_dir, 'best_model.pt')
    torch.save(best_weights, model_path)
    print(f'  Model weights  : {model_path}')

    # Save tokeniser files (vocabulary, config, special tokens)
    tok_path = os.path.join(artifacts_dir, 'tokenizer')
    tokenizer.save_pretrained(tok_path)
    print(f'  Tokeniser      : {tok_path}/')

    # Save all preprocessing parameters needed by predict()
    artifacts = {
        'speaker_cs_lookup'  : speaker_cs_lookup,
        'speaker_cred_lookup': cred_data['speaker_cred_lookup'],
        'affil_cred_lookup'  : cred_data['affil_cred_lookup'],
        'ctx_cred_lookup'    : cred_data['ctx_cred_lookup'],
        'len_mean'           : cred_data['len_mean'],
        'len_std'            : cred_data['len_std'],
        'default_cred'       : cred_data['default_cred'],
        'cs_default'         : _CS_DEFAULT,   # saved so predict.py can read it; fallback=0.5
        'best_threshold'     : best_t,
        'model_config'       : {
            'model_name': _MODEL_NAME,
            'hidden_dim': _HIDDEN_DIM,
            'dropout'   : _DROPOUT,
            'max_len'   : _MAX_LEN,
        },
    }
    pkl_path = os.path.join(artifacts_dir, 'preprocessors.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f'  Preprocessors  : {pkl_path}')

    print(f'\nAll artifacts saved to: {artifacts_dir}/')
    print('Next step: from truthfulness_classifier import predict')

    return artifacts
