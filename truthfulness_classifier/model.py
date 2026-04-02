"""
model.py — TruthClassifier neural network definition.

This file is imported by both train.py (for training) and predict.py (for inference).
Keeping the model definition in one place ensures both files always use the same
architecture — a mismatch would cause errors when loading saved weights.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel


class TruthClassifier(nn.Module):
    """
    Three-branch classifier for statement truthfulness.

    Architecture :
        Branch 1  text   : DistilBERT → [CLS] token vector  (768-dim)
        Branch 2  CS     : Linear(1→1) → tanh               (1-dim)
        Branch 3  meta   : [speaker_cred, affil_cred, ctx_cred, len_norm]  (4-dim)
        Head             : Linear(773→hidden_dim) → ReLU → Dropout → Linear(hidden_dim→1)

    Output: one raw logit per sample. Apply sigmoid() to get P(TRUE) in [0,1].

    Hyperparameters (Optuna best trial):
        hidden_dim = 256
        dropout    = 0.270
    """

    def __init__(self, model_name, hidden_dim, dropout):
        """
        Args:
            model_name : HuggingFace model ID, e.g. 'distilbert-base-uncased'
            hidden_dim : number of neurons in the hidden layer of the classifier head
            dropout    : dropout probability (applied after DistilBERT and inside head)
        """
        super().__init__()  # required by PyTorch — initialises nn.Module internals

        # Branch 1: DistilBERT encodes the statement text
        # DistilBertModel returns a tensor of shape (batch, max_len, 768)
        # We only use the [CLS] token (index 0) as a sentence-level representation
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        # Dropout randomly zeroes some activations during training to prevent overfitting
        self.bert_drop  = nn.Dropout(dropout)

        # Branch 2: Credit Score — learned linear scaling of  CS ratio
        # Linear(1, 1): one input (cs_ratio), one output (scaled cs)
        # tanh maps the output to (-1, 1) — applied in forward()
        self.cs_layer = nn.Linear(1, 1)

        # Classifier head: 768 (text) + 1 (CS) + 4 (metadata) = 773 inputs
        # Sequential applies the layers in order, left to right
        self.classifier = nn.Sequential(
            nn.Linear(773, hidden_dim),  # project from 773-dim to hidden_dim
            nn.ReLU(),                   # non-linearity: replace negatives with 0
            nn.Dropout(dropout),         # regularisation before final layer
            nn.Linear(hidden_dim, 1),   # output: one raw logit (NOT a probability)
        )

    def forward(self, input_ids, attention_mask, cs_ratio, metadata):
        """
        Forward pass: given inputs, compute one logit per sample.

        Args:
            input_ids      : (batch, max_len) integer token IDs from tokenizer
            attention_mask : (batch, max_len) 1 for real tokens, 0 for [PAD] tokens
            cs_ratio       : (batch, 1)  credit score in [0, 1] from Bhatt Eq.1
            metadata       : (batch, 4)  [speaker_cred, affil_cred, ctx_cred, len_norm]

        Returns:
            (batch, 1) raw logit — apply torch.sigmoid() to get P(TRUE)
        """
        # DistilBERT processes the token sequence
        # last_hidden_state: (batch, max_len, 768)
        bert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token is at position 0 — it's a learned summary of the whole sentence
        cls = self.bert_drop(bert_out.last_hidden_state[:, 0, :])  # (batch, 768)

        # Credit Score branch: apply learned linear transform then squash to (-1, 1)
        cs_out = torch.tanh(self.cs_layer(cs_ratio))               # (batch, 1)

        # Concatenate all three branches along the feature dimension (dim=1)
        # Result: (batch, 768 + 1 + 4) = (batch, 773)
        combined = torch.cat([cls, cs_out, metadata], dim=1)

        # Classifier head: (batch, 773) → (batch, 1)
        return self.classifier(combined)
