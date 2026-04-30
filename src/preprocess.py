import pandas as pd
import numpy as np


def load_data(filepath):
    """
    Loads raw CSV file and returns a DataFrame.

    Parameters (inputs):
    --------------------
    filepath : str
        Full or relative path to CSV file
        Example: '../data/raw/telco_churn.csv'

    Returns (outputs):
    ------------------
    df : pandas DataFrame
        Raw data exactly as loaded from CSV
    """

    df = pd.read_csv(filepath)

    print(f"Data loaded!")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    return df


def clean_data(df):
    """
    Performs all cleaning steps on raw DataFrame.

    Cleaning steps (in order):
    1. Copy data (protect original)
    2. Fix TotalCharges (text → number)
    3. Drop rows where TotalCharges is empty
    4. Drop customerID (not useful for prediction)
    5. Encode Churn column (Yes/No → 1/0)

    Parameters:
    -----------
    df : pandas DataFrame
        Raw data from load_data()

    Returns:
    --------
    df : pandas DataFrame
        Cleaned data with readable text labels      
        NOT yet one-hot encoded
        (kept readable for EDA visualizations!)
    """

    df = df.copy()


    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    rows_before = len(df)

    df = df.dropna(subset=['TotalCharges'])

    rows_dropped = rows_before - len(df)

    print(f"Fixed TotalCharges: dropped {rows_dropped} "
          f"empty rows")
    
    df = df.drop('customerID', axis=1)

    print(f"Dropped customerID column")

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print(f"Churn encoded: Yes→1, No→0")
    print(f"Cleaning done: {df.shape[0]} rows, "
          f"{df.shape[1]} columns")

    return df


def encode_features(df):
    """
    One-hot encodes all categorical (text) columns.

    WHY SEPARATE FROM clean_data()?
    --------------------------------
    clean_data() returns readable text labels
    → Perfect for EDA (charts show 'Male' not 1)

    encode_features() converts text to numbers
    → Perfect for ML model (needs numbers only)

    Keeping them separate gives us BOTH versions!

    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned data from clean_data()
        (must have text columns still intact)

    Returns:
    --------
    df_encoded : pandas DataFrame
        Fully numerical data ready for ML model

    feature_columns : list
        List of ALL column names after encoding
        CRITICAL for prediction later!
        New customer data must match these columns exactly!
    """

    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    print(f"   Found {len(categorical_cols)} "
          f"text columns to encode")
    print(f"   Columns: {categorical_cols}")


    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    feature_columns = [
        col for col in df_encoded.columns
        if col != 'Churn'
    ]

    print(f"Encoding done: {df_encoded.shape[1]} "
          f"total columns")
    print(f"   Feature columns: {len(feature_columns)}")

    return df_encoded, feature_columns


def run_preprocessing_pipeline(filepath):
    """
    Master function that runs the COMPLETE pipeline.

    This is the CONDUCTOR of the orchestra!
    Calls all other functions in correct order:
    Load → Clean → Encode → Save

    Why have this master function?
    → One function call does EVERYTHING
    → Correct order GUARANTEED
    → Used in notebook AND Streamlit app
    → No chance of running steps out of order!

    Parameters:
    -----------
    filepath : str
        Path to raw CSV file

    Returns:
    --------
    df_raw : DataFrame
        Original untouched data

    df_clean : DataFrame
        Cleaned with readable labels (for EDA!)

    df_encoded : DataFrame
        Fully numerical (for ML model!)

    feature_columns : list
        Column names for prediction validation
    """

    print("=" * 50)
    print("  PREPROCESSING PIPELINE STARTING")
    print("=" * 50)

    print("\n STEP 1: Loading data...")
    df_raw = load_data(filepath)
    print("\n STEP 2: Cleaning data...")
    df_clean = clean_data(df_raw)

    print("\n STEP 3: Encoding features...")
    df_encoded, feature_columns = encode_features(df_clean)

    print("\n STEP 4: Saving processed data...")
    output_path = '../data/processed/telco_churn_cleaned.csv'
    df_encoded.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    print("\n" + "=" * 50)
    print("  PIPELINE COMPLETE!")
    print("=" * 50)
    print(f"  Raw data:     {df_raw.shape[0]} rows, "
          f"{df_raw.shape[1]} cols")
    print(f"  Clean data:   {df_clean.shape[0]} rows, "
          f"{df_clean.shape[1]} cols")
    print(f"  Encoded data: {df_encoded.shape[0]} rows, "
          f"{df_encoded.shape[1]} cols")
    print(f"  Features:     {len(feature_columns)} columns")
    print("=" * 50)


    return df_raw, df_clean, df_encoded, feature_columns