"""Shared positional label masks for analyzer badges and exact row evidence."""
import numpy as np
import pandas as pd


def label_masks(series):
    values = series.astype('string')
    present = values.notna()
    normalized = values.str.strip().str.lower()
    counts = pd.DataFrame({'key': normalized, 'value': values}).dropna().groupby('key')['value'].nunique()
    ambiguous = counts[counts > 1].index
    return {
        'mixed_numeric': (present & pd.to_numeric(values, errors='coerce').isna()).to_numpy(dtype=bool),
        'equivalent': normalized.isin(ambiguous).to_numpy(dtype=bool),
        'whitespace': np.array([isinstance(v, str) and v != v.strip() for v in series], dtype=bool),
    }
