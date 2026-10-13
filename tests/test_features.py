import os
import json
import pickle
import sys
import numpy as np
import pandas as pd
import pytest  # → unconditional import; pytest is added to requirements.txt (T2)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import load_data, clean_data
from src.features import get_tenure_group, add_engineered_features, encode_for_inference


@pytest.mark.parametrize("t,expected", [                        # → one test case per tuple
    (0, 'New'), (12, 'New'), (13, 'Mid'), (48, 'Mid'), (49, 'Loyal'), (72, 'Loyal'),
])
def test_tenure_group(t, expected):                              # → pytest supplies t and expected
    assert get_tenure_group(t) == expected                       # → checks the shared helper


def test_single_row_encoding_parity():
    """
    Item 5 Parity Test:
    Picks one customer per Contract type ('Month-to-month', 'One year', 'Two year'),
    builds the single-row inference vector (as app/app.py does), and verifies that
    column order, encoded values, and scaler.transform() output match the batch
    training matrix row-for-row.
    """
    raw_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'telco_churn.csv')
    if not os.path.exists(raw_path):                                              # → CSV is gitignored, so may be absent
        pytest.skip('data/raw/telco_churn.csv not present (it is gitignored)')    # → skip, don't error

    models_dir = os.path.join(PROJECT_ROOT, 'models')

    with open(os.path.join(models_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, 'model_info.json'), 'r') as f:
        model_info = json.load(f)

    threshold = model_info.get('high_value_threshold', 70.6)

    df_clean = clean_data(load_data(raw_path))
    df_feat = add_engineered_features(df_clean, high_value_threshold=threshold)
    cat_cols = df_feat.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    df_batch = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True).drop('Churn', axis=1)

    for contract_type in ['Month-to-month', 'One year', 'Two year']:
        idx = df_clean[df_clean['Contract'] == contract_type].index[0]
        single_raw = df_clean.loc[[idx]].drop('Churn', axis=1)
        single_feat = add_engineered_features(single_raw, high_value_threshold=threshold)
        single_encoded = encode_for_inference(single_feat, feature_columns)

        expected_row = df_batch.loc[[idx], feature_columns].astype(float)

        assert list(single_encoded.columns) == list(expected_row.columns)
        np.testing.assert_allclose(single_encoded.values, expected_row.values)
        np.testing.assert_allclose(
            scaler.transform(single_encoded),
            scaler.transform(expected_row)
        )
