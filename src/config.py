# File Paths
DB_PATH = 'data/phishing.db'
SQL_QUERY = 'SELECT * FROM phishing_data'

# Target and Feature Columns
TARGET_COL = 'label'

# Features for specific processing
NUMERICAL_COLS = [
    'LargestLineLength', 'NoOfURLRedirect', 'NoOfPopup', 'NoOfiFrame', 
    'NoOfImage', 'NoOfSelfRef', 'NoOfExternalRef', 'LineOfCode', 'DomainAgeMonths'
]
BINARY_INT_COLS = ['Robots', 'IsResponsive']
CATEGORICAL_NOMINAL_COLS = ['Industry', 'HostingProvider']

# Model Selection
MODELS_TO_TRAIN = ['RandomForest', 'LogisticRegression', 'GradientBoosting']

# Model Hyperparameters (easy experimentation - change here!)
# Each parameter dictionary is unpacked using ** in train.py
MODEL_PARAMS = {
    'LogisticRegression': {
        'solver': 'liblinear',      # Efficient for small datasets
        'random_state': 42,          # Reproducibility
        'class_weight': 'balanced',  # Handle class imbalance
        'max_iter': 1000             # Maximum iterations for convergence
    },
    'RandomForest': {
        'n_estimators': 100,         # Number of decision trees
        'random_state': 42,          # Reproducibility
        'class_weight': 'balanced',  # Handle class imbalance
        'n_jobs': -1                 # Use all CPU cores for parallel training
    },
    'GradientBoosting': {
        'n_estimators': 100,         # Number of boosting stages
        'learning_rate': 0.1,        # Shrinkage parameter (0.01-0.3 typical)
        'random_state': 42           # Reproducibility
    }
}

# Train-Test Split Configuration
TEST_SIZE = 0.2      # 20% for testing, 80% for training
RANDOM_STATE = 42    # For reproducible splits

# Outlier Capping Configuration
OUTLIER_COLS = [
    'LargestLineLength', 'NoOfURLRedirect', 'NoOfPopup', 'NoOfiFrame',
    'NoOfImage', 'NoOfSelfRef', 'NoOfExternalRef'
]
