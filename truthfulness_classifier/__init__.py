"""
truthfulness_classifier — Binary truthfulness classifier for political statements.

Public API:
    train(csv_path, artifacts_dir)   → trains model, saves artifacts
    predict(input_dict, ...)         → {'prediction': 'TRUE'/'FALSE', 'explanation': ...}

Quick start:
    from truthfulness_classifier import predict

    result = predict({
        'statement'           : 'The unemployment rate has reached a record low.',
        'speaker'             : 'barack obama',
        'speaker_affiliation' : 'democrat',
        'context'             : 'a campaign speech',
    })
    print(result['prediction'])    # 'TRUE' or 'FALSE'
    print(result['explanation'])   # plain text explanation

To retrain from scratch:
    from truthfulness_classifier import train
    train('data.csv', artifacts_dir='distilbert_model')
"""

from .train import train
from .predict import predict

# __all__ controls what `from truthfulness_classifier import *` exports
__all__ = ['train', 'predict']
