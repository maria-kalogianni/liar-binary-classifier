# Truthfulness Classifier

Binary classification of statements as **TRUE** or **FALSE**, trained on the [LIAR dataset](https://huggingface.co/datasets/liar).

**Model:** DistilBERT + Credit Score + Metadata (speaker, affiliation, context)
**Test set:** Accuracy 0.714 · AUC 0.763 · MCC 0.418 · Balanced Accuracy 0.708

---

## Dataset

This project uses the **LIAR dataset** (Wang, 2017) — a benchmark dataset for fake news detection containing ~12.8K manually labelled short statements from PolitiFact, spanning six truthfulness classes: *pants-fire, false, barely-true, half-true, mostly-true, true*.

For this classifier the six classes are collapsed to binary labels: **TRUE** (mostly-true, true) and **FALSE** (pants-fire, false, barely-true, half-true).

> W. Y. Wang, "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection", ACL 2017.

---

## Quick start

```python
from truthfulness_classifier import predict

result = predict({
    'statement'           : 'The unemployment rate has reached a record low.',
    'speaker'             : 'barack obama',
    'speaker_affiliation' : 'democrat',
    'context'             : 'a campaign speech',
})

print(result['prediction'])    # 'TRUE' or 'FALSE'
print(result['explanation'])   # plain text explanation
```

All four fields are optional. Unknown speakers/contexts fall back to dataset averages.

---

## Installation

```bash
pip install -e .
```

Install all dependencies:

```bash
pip install torch transformers captum lime anthropic pandas scikit-learn numpy tqdm
```

---

## Environment variable

The explanation uses the Claude API (Haiku model). Set your key before running:

```bash
# Linux / macOS
export ANTHROPIC_API_KEY=sk-ant-...

# Windows
set ANTHROPIC_API_KEY=sk-ant-...
```

**If the key is not set**, `predict()` still works and returns a rule-based explanation
built directly from the attribution scores — no API call needed.

---

## Retrain from scratch

```python
from truthfulness_classifier import train

train('data.csv', artifacts_dir='distilbert_model')
```

Training saves three artifacts to `artifacts_dir/`:
- `best_model.pt` — model weights (best validation balanced accuracy)
- `tokenizer/` — DistilBERT tokenizer files
- `preprocessors.pkl` — credibility lookups, length stats, threshold


---

## File structure

```
truthfulness_classifier/
├── __init__.py      exposes train() and predict()
├── model.py         TruthClassifier (DistilBERT + CS + metadata)
├── preprocess.py    deduplication, normalization, feature engineering
├── train.py         full training pipeline
├── explainer.py     LayerIntegratedGradients + GradientSHAP + Claude API
└── predict.py       inference entry point (singleton model loading)
pyproject.toml
distilbert_model/    saved artifacts (not in git)
```

---

## predict() — input / output

**Input dict keys:**

| Key                    | Required | Example                    |
|------------------------|----------|----------------------------|
| `statement`            | yes      | `"Taxes increased by 30%"` |
| `speaker`              | no       | `"barack obama"`           |
| `speaker_affiliation`  | no       | `"democrat"`               |
| `context`              | no       | `"a chain email"`          |

**Output dict:**

| Key           | Type   | Example                        |
|---------------|--------|--------------------------------|
| `prediction`  | str    | `"TRUE"` or `"FALSE"`         |
| `explanation` | str    | plain text, 2-3 sentences      |

---

## Performance (test set, deduped, threshold = 0.48)

| Metric            | Value |
|-------------------|-------|
| Accuracy          | 0.713 |
| Balanced Accuracy | 0.708 |
| AUC-ROC           | 0.763 |
| MCC               | 0.418 |
| F1                | 0.727 |
| Precision         | 0.774 |
| Recall            | 0.685 |






