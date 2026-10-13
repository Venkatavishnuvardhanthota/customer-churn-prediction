import pandas as pd

SERVICE_COLS = [
    'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies'
]


def get_tenure_group(t: int) -> str:
    if t <= 12:
        return 'New'
    elif t <= 48:
        return 'Mid'
    else:
        return 'Loyal'


def add_engineered_features(df: pd.DataFrame, high_value_threshold: float = None) -> pd.DataFrame:
    """
    Adds the 5 engineered features to a cleaned customer DataFrame.
    If high_value_threshold is None, high_value_customer is initialized to 0
    so the threshold can be fit on X_train after train_test_split.
    """
    out = df.copy()
    out['tenure_group'] = pd.cut(
        out['tenure'],
        bins=[-1, 12, 48, 72],
        labels=['New', 'Mid', 'Loyal']
    )
    out['total_services'] = (out[SERVICE_COLS] == 'Yes').sum(axis=1)
    out['is_new_customer'] = (out['tenure'] <= 12).astype(int)
    out['avg_monthly_from_total'] = out['TotalCharges'] / out['tenure'].replace(0, 1)
    if high_value_threshold is not None:
        out['high_value_customer'] = (out['MonthlyCharges'] > high_value_threshold).astype(int)
    else:
        out['high_value_customer'] = 0
    return out


def encode_for_inference(input_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Encodes a single-row or multi-row DataFrame with engineered features
    to match the exact training feature_columns order and values.
    """
    categorical_cols = input_df.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded.astype(float)
