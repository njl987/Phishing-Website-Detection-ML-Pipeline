from src.data_loader import load_data
from src.preprocessing import clean_data, preprocess_features, split_data
from src.train import train_models
from src.evaluate import evaluate_models, log_results

def main():
    """ Runs the complete ML pipeline. """
    print("--- Starting ML Pipeline Execution ---")

    # 1. Data Loading (Requires data/phishing.db to exist)
    try:
        data = load_data()
    except Exception:
        print("Pipeline aborted due to data loading error.")
        return

    # 2. Data Cleaning (Based on EDA findings)
    # Handles missing values, outliers, data contamination, categorical standardization
    cleaned_data = clean_data(data)

    # 3. Feature Preprocessing (OHE and Standardization)
    # One-Hot Encoding for categories, StandardScaler for numerical features
    X, y = preprocess_features(cleaned_data)
    
    # 4. Data Splitting (80/20 train-test split, stratified to preserve class balance)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Model Training (Trains 3 selected models: RF, LR, GB)
    trained_models = train_models(X_train, y_train)

    # 6. Model Evaluation (Calculates Accuracy, Precision, Recall, F1 + Confusion Matrix)
    results = evaluate_models(trained_models, X_test, y_test)
    
    # 7. Logging Results
    log_results(results)

    print("--- ML Pipeline Execution Finished Successfully ---")

if __name__ == '__main__':
    main()
