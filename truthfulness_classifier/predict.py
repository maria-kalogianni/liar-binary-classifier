"""
predict.py — Inference entry point for TruthClassifier.

Entry point: predict(input_dict, artifacts_dir)

What it does:
    1. Load model + tokenizer + preprocessors (only on FIRST call — then cached)
    2. Clean and normalise the input statement and speaker fields
    3. Look up CS ratio and credibility scores from saved artifacts
    4. Run model forward pass → P(TRUE) → 'TRUE' or 'FALSE'
    5. Generate explanation via Explainer (LayerIG + GradientSHAP + Claude API)
    6. Return {'prediction': ..., 'explanation': ...}

"""

import os
import pickle
import warnings
import torch
from transformers import DistilBertTokenizerFast

from .preprocess import standardize_format, norm_speaker, norm_context, norm_affil
from .model import TruthClassifier
from .explainer import Explainer

warnings.filterwarnings('ignore')


# ── Module-level singletons ───────────────────────────────────────────────────
# These are loaded ONCE on the first call to predict() and reused for every
# subsequent call. Loading DistilBERT takes ~5 seconds — we must not reload it
# on every prediction.

_model     = None    # TruthClassifier instance
_tokenizer = None    # DistilBertTokenizerFast instance
_artifacts = None    # dict from preprocessors.pkl
_explainer = None    # Explainer instance
_device    = None    # torch.device (cpu or cuda)

# Normalised lookup dictionaries — built once from pkl keys at load time.
# We pre-normalise the keys so predict() does not need to re-normalise on every call.
_SPEAKER_CS_NORM   = None   # {norm_speaker(name): cs_ratio}
_SPEAKER_CRED_NORM = None   # {norm_speaker(name): TRUE_rate}
_AFFIL_CRED_NORM   = None   # {norm_affil(affil): TRUE_rate}
_CTX_CRED_NORM     = None   # {norm_context(ctx): TRUE_rate}

# CS_DEFAULT: fallback credit score for unknown speakers.
# New artifacts (from train.py) include 'cs_default' in the pkl.
# Old artifacts (Day 6 saved weights) do not — so we keep 0.5 as a module-level
# fallback and override it with the pkl value inside _load_artifacts().
CS_DEFAULT = 0.5   # overridden at load time if pkl contains 'cs_default' key


def _load_artifacts(artifacts_dir):
    """
    Load and cache all model artifacts.
    Called automatically on the first predict() call.

    Args:
        artifacts_dir : folder containing best_model.pt, tokenizer/, preprocessors.pkl

    Sets all module-level singletons (_model, _tokenizer, _artifacts, etc.)
    """
    global _model, _tokenizer, _artifacts, _explainer, _device
    global _SPEAKER_CS_NORM, _SPEAKER_CRED_NORM, _AFFIL_CRED_NORM, _CTX_CRED_NORM

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[predict] Loading artifacts from: {artifacts_dir}/')
    print(f'[predict] Device: {_device}')

    # ── Load preprocessors.pkl ────────────────────────────────────────────────
    pkl_path = os.path.join(artifacts_dir, 'preprocessors.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f'preprocessors.pkl not found at: {pkl_path}\n'
            f'Run train() first, or point artifacts_dir to an existing distilbert_model/ folder.'
        )
    with open(pkl_path, 'rb') as f:
        _artifacts = pickle.load(f)

    # Override module-level CS_DEFAULT with the value stored in pkl (if present).
    # New artifacts (produced by train.py) include 'cs_default'.
    # Old artifacts (Day 6 saved weights) do not — in that case keep the 0.5 fallback.
    global CS_DEFAULT
    CS_DEFAULT = _artifacts.get('cs_default', CS_DEFAULT)

    # ── Pre-normalise lookup dictionary keys ──────────────────────────────────
    # The pkl keys are in raw CSV format (e.g. 'barack-obama' for speakers).
    # We normalise them once here so every call to predict() can do a fast dict lookup
    # without re-normalising.
    #
    # Edge case — duplicate normalised keys:
    #   If two raw keys happen to normalise to the same string, the second one
    #   silently overwrites the first in the dict. Example (hypothetical):
    #     'John Smith' and 'john-smith' both → norm_speaker → 'john-smith'
    #   This is extremely unlikely for speaker names in the dataset (which
    #   already stores names in 'first-last' format). For contexts and affiliations
    #   the risk is also negligible — their normalization only strips + lowercases.
    #   If it were a concern, we could check with: assert len(result) == len(raw_dict)
    _SPEAKER_CS_NORM   = {
        norm_speaker(k): v for k, v in _artifacts.get('speaker_cs_lookup', {}).items()
    }
    _SPEAKER_CRED_NORM = {
        norm_speaker(k): v for k, v in _artifacts.get('speaker_cred_lookup', {}).items()
    }
    _AFFIL_CRED_NORM   = {
        norm_affil(k): v for k, v in _artifacts.get('affil_cred_lookup', {}).items()
    }
    _CTX_CRED_NORM     = {
        norm_context(k): v for k, v in _artifacts.get('ctx_cred_lookup', {}).items()
    }
    print(f'[predict] Lookups: {len(_SPEAKER_CS_NORM)} speakers, '
          f'{len(_CTX_CRED_NORM)} contexts, {len(_AFFIL_CRED_NORM)} affiliations')

    # ── Load model ────────────────────────────────────────────────────────────
    cfg = _artifacts.get('model_config', {
        'model_name': 'distilbert-base-uncased',
        'hidden_dim': 256,
        'dropout'   : 0.27,
        'max_len'   : 128,
    })
    _model = TruthClassifier(
        model_name=cfg['model_name'],
        hidden_dim=cfg['hidden_dim'],
        dropout=cfg['dropout'],
    ).to(_device)

    # Load saved weights
    # weights_only=False: required for compatibility with checkpoints saved before
    # PyTorch 2.6 (which made weights_only=True the default).
    model_path = os.path.join(artifacts_dir, 'best_model.pt')
    try:
        state_dict = torch.load(model_path, map_location=_device, weights_only=False)
    except TypeError:
        # PyTorch < 1.13 does not have the weights_only parameter
        state_dict = torch.load(model_path, map_location=_device)

    _model.load_state_dict(state_dict)

    # IMPORTANT: eval() disables Dropout for deterministic inference.
    # Without eval(), each call to predict() could give slightly different results.
    _model.eval()
    print('[predict] Model loaded.')

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tok_path = os.path.join(artifacts_dir, 'tokenizer')
    try:
        _tokenizer = DistilBertTokenizerFast.from_pretrained(tok_path)
        print('[predict] Tokenizer loaded from saved files.')
    except Exception:
        # Fallback: download from HuggingFace if saved files are missing
        _tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        print('[predict] Tokenizer downloaded from HuggingFace (saved files not found).')

    # ── Create Explainer (attaches Captum attributors) ────────────────────────
    _explainer = Explainer(_model, _tokenizer, _artifacts, _device)
    print('[predict] Ready. Call predict(input_dict) to classify a statement.\n')


def _prepare_input(input_dict):
    """
    Convert a raw input dict into tensors ready for the model.

    Normalisation rules (field-specific — see preprocess.py for rationale):
        speaker name    → norm_speaker() → spaces become hyphens
        context         → norm_context() → lowercase + strip only
        affiliation     → norm_affil()   → lowercase + strip only

    For each field, we look up the value in the pre-normalised dict.
    If not found, we fall back to the default credibility value.

    Args:
        input_dict : dict with optional keys:
                     'statement', 'speaker', 'speaker_affiliation', 'context'

    Returns:
        dict with tensors and raw scalar values needed by Explainer
    """
    cfg          = _artifacts.get('model_config', {'max_len': 128})
    default_cred = _artifacts.get('default_cred', 0.557)
    len_mean     = _artifacts.get('len_mean', 50.0)
    len_std      = _artifacts.get('len_std', 20.0)
    max_len      = cfg.get('max_len', 128)

    # Extract and clean inputs — use empty string if field is missing.
    # Each field accepts both the data.csv column name AND a shorter alias,
    # so that the caller can pass a raw data.csv row without renaming keys.
    #   data.csv name       short alias used in earlier examples
    #   ─────────────────── ──────────────────────────────────────
    #   speaker_name      → speaker
    #   statement_context → context
    # Fields not used by the model (subjects, speaker_job, speaker_state)
    # are accepted silently and ignored.
    statement = standardize_format(input_dict.get('statement', ''))
    speaker   = str(input_dict.get('speaker',
                    input_dict.get('speaker_name', ''))).strip()
    affil     = str(input_dict.get('speaker_affiliation', '')).strip()
    context   = str(input_dict.get('context',
                    input_dict.get('statement_context', ''))).strip()

    # Tokenise the statement
    enc = _tokenizer(
        statement,
        max_length=max_len,
        padding='max_length',    # pad short sequences with [PAD] tokens
        truncation=True,         # cut sequences longer than max_len
        return_tensors='pt',     # return PyTorch tensors
    )
    input_ids      = enc['input_ids'].to(_device)        # (1, max_len)
    attention_mask = enc['attention_mask'].to(_device)   # (1, max_len)

    # ── Credit Score lookup ───────────────────────────────────────────────────
    # Separately track cs_known so the Explainer can skip cs_ratio in the prompt
    # for unknown speakers (their CS = default = 0.5 = uninformative).
    speaker_key = norm_speaker(speaker)
    cs_val_or_none = _SPEAKER_CS_NORM.get(speaker_key)   # None if not found

    if cs_val_or_none is None:
        cs_val   = CS_DEFAULT
        cs_known = False    # speaker not in training data → attribution is uninformative
    else:
        cs_val   = cs_val_or_none
        cs_known = True

    # ── Credibility lookups ───────────────────────────────────────────────────
    spk_cred = _SPEAKER_CRED_NORM.get(norm_speaker(speaker), default_cred)
    afl_cred = _AFFIL_CRED_NORM.get(norm_affil(affil),       default_cred)
    ctx_cred = _CTX_CRED_NORM.get(norm_context(context),     default_cred)

    # Z-score normalise statement length: (chars - mean) / std
    len_norm = (len(statement) - len_mean) / len_std

    # Build tensors
    cs_tensor   = torch.tensor([[cs_val]], dtype=torch.float32).to(_device)  # (1, 1)
    meta_tensor = torch.tensor(
        [[spk_cred, afl_cred, ctx_cred, len_norm]], dtype=torch.float32
    ).to(_device)  # (1, 4)

    return {
        # Tensors for model + Captum
        'statement'     : statement,
        'input_ids'     : input_ids,
        'attention_mask': attention_mask,
        'cs_tensor'     : cs_tensor,
        'meta_tensor'   : meta_tensor,
        # Scalars for Explainer prompt building
        'cs_val'        : cs_val,
        'cs_known'      : cs_known,
        'spk_cred'      : spk_cred,
        'afl_cred'      : afl_cred,
        'ctx_cred'      : ctx_cred,
        'len_norm'      : len_norm,
        'speaker'       : speaker,
        'context'       : context,
    }


def predict(input_dict, artifacts_dir='distilbert_model', n_steps=20, n_samples=50):
    """
    Classify a political statement as TRUE or FALSE and explain the decision.

    Args:
        input_dict    : dict with keys matching data.csv columns (all optional
                        except 'statement'). Accepts both data.csv names and
                        short aliases:
                          'statement'                      (required)
                          'speaker_name'  or 'speaker'     (optional)
                          'speaker_affiliation'            (optional)
                          'statement_context' or 'context' (optional)
                          'subjects', 'speaker_job', 'speaker_state'
                                                           (accepted, ignored)
        artifacts_dir : folder with best_model.pt, tokenizer/, preprocessors.pkl
                        (default: 'distilbert_model' relative to current directory)
        n_steps       : LayerIG integration steps (20 for CPU, 50 for GPU)
        n_samples     : GradientSHAP samples (50 for CPU, 100 for GPU)

    Returns:
        dict:
            'prediction'  : 'TRUE' or 'FALSE'
            'explanation' : plain text string explaining the decision

    Example:
        from truthfulness_classifier import predict

        result = predict({
            'statement'           : 'The unemployment rate has reached a record low.',
            'speaker'             : 'barack obama',
            'speaker_affiliation' : 'democrat',
            'context'             : 'a campaign speech',
        })
        print(result['prediction'])    # 'TRUE' or 'FALSE'
        print(result['explanation'])   # 2-3 sentence plain text explanation

    Expected output when running successfully:
        [predict] Loading artifacts from: distilbert_model/
        [predict] Device: cpu
        [predict] Lookups: 2871 speakers, 1183 contexts, 13 affiliations
        [predict] Model loaded.
        [predict] Tokenizer loaded from saved files.
        [predict] Ready. Call predict(input_dict) to classify a statement.
        [predict] TRUE  (prob=0.623, threshold=0.57)
          LayerIG (20 steps)... done.
          IG convergence: ACCEPTABLE (delta=0.0842)
          GradientSHAP (50 samples)... done.
        {'prediction': 'TRUE', 'explanation': '...'}
    """
    # Load all artifacts on the FIRST call (singleton pattern)
    # Subsequent calls skip this block and reuse the cached objects
    if _model is None:
        _load_artifacts(artifacts_dir)

    # Convert raw input dict to model-ready tensors
    prep = _prepare_input(input_dict)

    # ── Model inference ───────────────────────────────────────────────────────
    with torch.no_grad():   # no gradient computation needed during inference
        logit = _model(
            prep['input_ids'],
            prep['attention_mask'],
            prep['cs_tensor'],
            prep['meta_tensor'],
        )

    # sigmoid converts logit to probability in [0, 1]
    prob      = torch.sigmoid(logit).item()
    threshold = _artifacts.get('best_threshold', 0.57)
    label     = 'TRUE' if prob >= threshold else 'FALSE'

    print(f'[predict] {label}  (prob={prob:.3f}, threshold={threshold:.2f})')

    # ── Generate explanation ──────────────────────────────────────────────────
    explanation = _explainer.explain(
        prep, prob, label,
        n_steps=n_steps,
        n_samples=n_samples,
    )

    return {
        'prediction' : label,
        'explanation': explanation,
    }
