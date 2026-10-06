"""
preprocessing.py - Data loading, audit, cleaning, and sklearn preprocessing pipeline building.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw Telco Customer Churn CSV dataset."""
    df = pd.read_csv(filepath)
    return df


def audit_data(df: pd.DataFrame) -> dict:
    """Audit dataset dimensions, column types, missing values, duplicates, and churn rate."""
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_counts = df.isnull().sum().to_dict()
    
    # Check for blank space missing values in TotalCharges
    total_charges_blanks = 0
    if 'TotalCharges' in df.columns:
        total_charges_blanks = (df['TotalCharges'].astype(str).str.strip() == '').sum()
    
    churn_dist = {}
    churn_rate = 0.0
    if 'Churn' in df.columns:
        churn_dist = df['Churn'].value_counts().to_dict()
        churn_yes = churn_dist.get('Yes', churn_dist.get(1, 0))
        churn_rate = churn_yes / total_rows if total_rows > 0 else 0.0

    return {
        "shape": (total_rows, total_cols),
        "duplicates": int(df.duplicated().sum()),
        "missing_counts": missing_counts,
        "total_charges_blanks": int(total_charges_blanks),
        "churn_distribution": churn_dist,
        "churn_rate_pct": round(churn_rate * 100, 2)
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Convert TotalCharges string blanks ' ' to numeric np.nan and fill with median
    - Convert binary Churn target to 1/0
    """
    df_clean = df.copy()

    # Convert TotalCharges to numeric
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = df_clean['TotalCharges'].astype(str).str.strip()
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        # Impute missing TotalCharges with median
        median_total = df_clean['TotalCharges'].median()
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_total if pd.notnull(median_total) else 0.0)

    # Convert target Churn to binary 1/0 if string
    if 'Churn' in df_clean.columns:
        if df_clean['Churn'].dtype == 'object':
            df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    return df_clean


def get_feature_target_split(df: pd.DataFrame, target_col: str = 'Churn'):
    """Separate feature matrix X and target vector y, dropping non-predictive ID column."""
    df_modeling = df.copy()
    
    if 'customerID' in df_modeling.columns:
        df_modeling = df_modeling.drop(columns=['customerID'])

    X = df_modeling.drop(columns=[target_col])
    y = df_modeling[target_col]

    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construct reproducible ColumnTransformer for numeric and categorical features.
    - Numerical: SimpleImputer (median) + StandardScaler
    - Categorical: SimpleImputer (most_frequent) + OneHotEncoder (drop='first', handle_unknown='ignore')
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure SeniorCitizen is treated as categorical if present in numeric_features
    if 'SeniorCitizen' in numeric_features:
        numeric_features.remove('SeniorCitizen')
        categorical_features.append('SeniorCitizen')

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def prepare_stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """Split dataset reproducibly into train and test sets using stratification."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
