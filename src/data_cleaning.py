"""
data_cleaning.py
================
Student Performance Prediction — Data Cleaning & Preprocessing Module
Decodelabs Data Science Internship — Task 2

All cleaning logic lives here as reusable functions.
The notebook imports and calls these functions — keeping
the notebook clean and the logic easy to test.

Usage:
    from src.data_cleaning import load_raw_data, clean_data, get_summary
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


#  Column definitions 

BINARY_COLS = [
    'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]

MULTI_CAT_COLS = [
    'school', 'sex', 'address', 'famsize',
    'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian'
]

SCALE_COLS = [
    'age', 'absences', 'G1', 'G2',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'parent_edu_avg', 'alc_total', 'grade_avg_g1g2'
]

#  Step 1: Load 
def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw student-mat.csv file.

    This dataset uses semicolon separators (European format).
    G1 and G2 are stored as quoted strings in the raw file —
    we convert them to numeric here.

    Args:
        path: Path to student-mat.csv

    Returns:
        Raw DataFrame with 395 rows × 33 columns
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, sep=';')

    # G1/G2 come in as strings due to quotes in raw file
    df['G1'] = pd.to_numeric(df['G1'], errors='coerce')
    df['G2'] = pd.to_numeric(df['G2'], errors='coerce')

    print(f"Raw data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


#  Step 2: Standardise string columns 

def standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    str_cols = df.select_dtypes(include=['object']).columns

    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    print(f"Standardised {len(str_cols)} string columns (strip + lowercase)")
    return df



#  Step 3: Remove duplicates 

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows. Duplicates bias the model by
    making it learn from the same student more than once.
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")
    else:
        print(f"No duplicates found")
    return df



#  Step 4: Handle missing values 

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and handle missing values.

    Strategy per column type:
    - Numerical  → fill with median (robust to outliers)
    - Categorical → fill with mode  (most frequent value)

    This dataset has no missing values, but the function
    handles any that appear after loading.
    """
    df = df.copy()
    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        print(f"No missing values found — nothing to fill")
        return df

    num_cols = df.select_dtypes(include='number').columns
    cat_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col].dtype)]

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"{col}: filled {df[col].isnull().sum()} nulls with median ({median_val})")

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"{col}: filled {df[col].isnull().sum()} nulls with mode ({mode_val})")

    print(f"Missing values handled — {total_missing} cells filled")
    return df



#  Step 5: Flag dropout students 

def flag_dropouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Students with G2=0 AND G3=0 likely dropped out mid-year —
    their final grade is 0 not because they failed but because
    they never sat the exam.

    We flag them rather than remove them, giving the model a
    chance to learn this pattern.
    """
    df = df.copy()
    dropout_mask = (df['G3'] == 0) & (df['G2'] == 0)
    df['is_dropout'] = dropout_mask.astype(int)
    count = dropout_mask.sum()
    print(f"Flagged {count} likely dropout students (G2=0 & G3=0) -> 'is_dropout' column")
    return df



#  Step 6: Encode binary columns 

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yes/no columns to 1/0.
    Machine learning models require numeric input — this is the
    simplest encoding for two-value categorical columns.
    """
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0})
    print(f"Encoded {len(BINARY_COLS)} binary columns (yes->1, no->0)")
    return df


#  Step 7: Encode multi-category columns 

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label encode multi-value categorical columns.

    Why Label Encoding here:
    - Columns like school (GP/MS) and sex (F/M) have no order,
      but with only 2–5 unique values, label encoding works well
      for tree-based models and is simpler than one-hot for this
      project size.

    Returns:
        DataFrame with encoded columns + encoding_map dict stored
        as df.attrs for reference.
    """
    df = df.copy()
    encoding_map = {}
    le = LabelEncoder()

    for col in MULTI_CAT_COLS:
        if col in df.columns:
            original_values = df[col].unique().tolist()
            df[col] = le.fit_transform(df[col])
            encoded_values = sorted(df[col].unique().tolist())
            encoding_map[col] = dict(zip(
                sorted(original_values),
                encoded_values
            ))

    df.attrs['encoding_map'] = encoding_map
    print(f"Label encoded {len(MULTI_CAT_COLS)} categorical columns")
    return df

#  Step 8: Cap outliers 

def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme outliers in the 'absences' column only.

    Why absences only:
    - All other numerical columns (grades, ratings) have fixed
      valid ranges (0-20, 1-5) — capping would distort real values.
    - Absences has a long tail (some students: 75 absences) that
      would skew model training without adding information.

    Strategy: Winsorize at 95th percentile.
    """
    df = df.copy()
    cap_value = df['absences'].quantile(0.95)
    original_max = df['absences'].max()
    df['absences'] = df['absences'].clip(upper=cap_value)
    print(f"Absences capped: {original_max} -> {cap_value} (95th percentile)")
    return df


#  Step 9: Feature engineering 

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 3 new meaningful features from existing columns.

    New features:
    - parent_edu_avg  : Average of mother + father education level.
                        Combined parental education is a stronger
                        predictor than either alone.
    - alc_total       : Workday + weekend alcohol consumption combined.
                        Total alcohol impact on performance.
    - grade_avg_g1g2  : Average of G1 and G2.
                        Strong predictor of G3 without being G3 itself.
    """
    df = df.copy()
    df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
    df['alc_total']      = df['Dalc'] + df['Walc']
    df['grade_avg_g1g2'] = (df['G1'] + df['G2']) / 2
    print(f"Created 3 new features: parent_edu_avg, alc_total, grade_avg_g1g2")
    return df

#  Step 10: Add target column 

def add_target_column(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """
    Create binary pass_fail target column.

    Portuguese grading: 10/20 is the minimum passing grade.
    pass_fail = 1 → Pass (G3 >= threshold)
    pass_fail = 0 → Fail (G3 < threshold)
    """
    df = df.copy()
    df['pass_fail'] = (df['G3'] >= threshold).astype(int)
    pass_n = df['pass_fail'].sum()
    fail_n = len(df) - pass_n
    print(f"Target column 'pass_fail' added")
    print(f"     Pass: {pass_n} ({pass_n/len(df)*100:.1f}%)  |  Fail: {fail_n} ({fail_n/len(df)*100:.1f}%)")
    return df


#  Step 11: Scale features 

def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise continuous numerical features using StandardScaler.
    (mean=0, std=1)

    Why scale:
    - Required for Logistic Regression and distance-based models.
    - Does NOT affect tree-based models (Random Forest) but doesn't
      hurt them either.
    - We scale a copy and keep the original unscaled for EDA.

    Returns:
        Tuple of (scaled_df, fitted_scaler)
        Keep the scaler — you need it to transform new data at
        prediction time.
    """
    df = df.copy()
    scaler = StandardScaler()
    cols_to_scale = [c for c in SCALE_COLS if c in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print(f"Scaled {len(cols_to_scale)} numerical features (mean=0, std=1)")
    return df, scaler



#  Master pipeline 

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all cleaning steps in order.
    Returns cleaned (but unscaled) DataFrame.
    Scaling is separate because EDA needs the original scale.
    """
    print("\n Starting data cleaning pipeline...")
    print("-" * 50)
    df = standardise_strings(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = flag_dropouts(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = cap_outliers(df)
    df = engineer_features(df)
    df = add_target_column(df)
    print("-" * 50)
    print(f"Cleaning complete: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


#  Summary report 

def get_cleaning_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Print a before/after summary of the cleaning process."""
    print("\n" + "=" * 55)
    print("  TASK 2 SUMMARY — Data Cleaning & Preprocessing")
    print("=" * 55)
    print(f"""
  Before cleaning
    Shape          : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns
    Missing values : {df_raw.isnull().sum().sum()}
    Duplicates     : {df_raw.duplicated().sum()}
    String columns : {len([col for col in df_raw.columns if pd.api.types.is_string_dtype(df_raw[col].dtype)])}

  After cleaning
    Shape          : {df_clean.shape[0]} rows × {df_clean.shape[1]} columns
    Missing values : {df_clean.isnull().sum().sum()}
    All numeric    : {all(df_clean.dtypes != 'object')}
    New features   : parent_edu_avg, alc_total, grade_avg_g1g2
    New columns    : is_dropout, pass_fail

  Encoding applied
    Binary (yes/no): yes→1, no→0  ({8} columns)
    Label encoded  : school, sex, address, famsize, Pstatus,
                     Mjob, Fjob, reason, guardian

  Outlier treatment
    absences       : capped at 95th percentile

  Target variable
    pass_fail      : {df_clean['pass_fail'].sum()} pass | {(df_clean['pass_fail']==0).sum()} fail
""")
    print("=" * 55)
