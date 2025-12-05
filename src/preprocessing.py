import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.config import NUMERICAL_COLS, BINARY_INT_COLS, CATEGORICAL_NOMINAL_COLS, TARGET_COL, OUTLIER_COLS, TEST_SIZE, RANDOM_STATE

def clean_data(df):
    """ Performs all data cleaning steps as per the EDA findings. """
    print("Starting data cleaning...")

    # 1. Drop redundant column
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("✅ 'Unnamed: 0' column dropped if it existed.")

    # 2. Impute missing values in 'LineOfCode' with median
    if 'LineOfCode' in df.columns:
        median_loc = df['LineOfCode'].median()
        df['LineOfCode'] = df['LineOfCode'].fillna(median_loc)
        print(f"✅ Missing values in 'LineOfCode' imputed with median: {median_loc:.2f}")

    # 3. Correct invalid values in 'NoOfImage' (set negatives to 0)
    if 'NoOfImage' in df.columns:
        df['NoOfImage'] = df['NoOfImage'].apply(lambda x: max(0, x))
        print("✅ Negative values in 'NoOfImage' corrected to 0.")

    # 4. Standardize categorical features (lowercase, strip)
    for col in CATEGORICAL_NOMINAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            print(f"✅ Standardized categorical column: {col}")
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    # 5. Convert binary features to int, robust to non-numeric values
    for col in BINARY_INT_COLS:
        if col in df.columns:
            # Coerce non-numeric to NaN, then fill with mode
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                mode = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode)
                print(f"✅ Filled non-numeric values in '{col}' with mode: {mode}")
            df[col] = df[col].astype(int)
            print(f"✅ Converted binary column '{col}' to int.")

    # 6. Outlier capping using IQR method (Interquartile Range)
    # Formula: Values beyond Q1-1.5*IQR or Q3+1.5*IQR are capped to those bounds
    for col in OUTLIER_COLS:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)  # 25th percentile
            Q3 = df[col].quantile(0.75)  # 75th percentile
            IQR = Q3 - Q1  # Interquartile range
            lower_bound = Q1 - 1.5 * IQR  # Lower fence for outliers
            upper_bound = Q3 + 1.5 * IQR  # Upper fence for outliers
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)  # Cap extreme values
            print(f"✅ Outliers in '{col}' capped using IQR method. Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")

    # 7. Detect and fix data contamination in numeric columns
    # Purpose: Find text/symbols in columns that should be numbers (e.g., '?' or 'unknown')
    numeric_cols = NUMERICAL_COLS + BINARY_INT_COLS
    for col in numeric_cols:
        if col in df.columns:
            # Check if column has non-numeric values (coerce converts invalid to NaN)
            non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
            if non_numeric_mask.any():
                contaminated_values = df.loc[non_numeric_mask, col].unique()
                print(f"Warning: Data contamination detected in '{col}': {contaminated_values}")
                # Replace contaminated values with NaN, then impute appropriately
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col in BINARY_INT_COLS:
                    # Binary columns: use most frequent value (mode)
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                else:
                    # Continuous columns: use median (robust to outliers)
                    fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                print(f"✅ Contaminated values in '{col}' replaced with {fill_value}")
    
    print("Data cleaning complete.")
    return df

def preprocess_features(df):
    """ Applies feature engineering (encoding, scaling) and creates X and y. """
    print("Starting feature engineering (encoding and scaling)...")

    # 1. Categorical Features Encoded (One-Hot Encoding)
    # Convert text categories (e.g., 'Industry', 'HostingProvider') into binary columns
    existing_cats = [col for col in CATEGORICAL_NOMINAL_COLS if col in df.columns]
    if existing_cats:
        # dummy_na=False: Don't create separate column for missing values (already handled in cleaning)
        df = pd.get_dummies(df, columns=existing_cats, prefix=existing_cats, dummy_na=False)
        print(f"✅ One-hot encoded columns: {existing_cats}")
    else:
        print("No categorical columns to encode.")

    # Ensure all columns are numeric before creating feature matrix
    # Safety check: Convert any remaining non-numeric values to 0
    for col in df.columns:
        if col != TARGET_COL:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Create the feature matrix X and target vector y
    X = df.drop(TARGET_COL, axis=1, errors='ignore')
    y = df[TARGET_COL]

    # 2. Standardization of Continuous Features
    # Transform features to mean=0, std=1 (important for LogisticRegression, less critical for tree models)
    cols_to_scale = [col for col in X.columns if col in NUMERICAL_COLS]

    if cols_to_scale:
        scaler = StandardScaler()  # Fit scaler on training data statistics
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        print(f"StandardScaled continuous features: {cols_to_scale}")
        
        # Save scaler for deployment
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        print("✅ Scaler saved to models/scaler.pkl")
    
    print("Feature preprocessing complete.")
    return X, y

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """ Splits the feature and target data into training and testing sets. """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Data split: Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test
