from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def evaluate_models(trained_models, X_test, y_test):
    """ Evaluates models and calculates key performance metrics. """
    results = {}
    
    print("Starting model evaluation...")
    for name, model in trained_models.items():
        # Generate predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate key metrics (all range from 0.0 to 1.0, higher is better)
        accuracy = accuracy_score(y_test, y_pred)   # Overall correctness
        precision = precision_score(y_test, y_pred) # Of predicted phishing, how many are actually phishing?
        recall = recall_score(y_test, y_pred)       # Of actual phishing, how many did we catch?
        f1 = f1_score(y_test, y_pred)               # Harmonic mean of precision & recall               # Harmonic mean of precision & recall
        
        # Confusion Matrix: [[TN, FP], [FN, TP]] - shows prediction breakdown
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ConfusionMatrix': cm.tolist()
        }
        
        print(f"\n--- {name} Results ---")
        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:")
        print(cm)

    return results

def log_results(results):
    """ Saves the evaluation results to a CSV file and prints summary. """
    print("\nLogging final results...")
    
    # Create a DataFrame for metric comparison (exclude ConfusionMatrix - too large for CSV)
    metrics_data = {
        model: {k: v for k, v in metrics.items() if k != 'ConfusionMatrix'} 
        for model, metrics in results.items()
    }
    # Transpose: models as rows, metrics as columns
    metrics_df = pd.DataFrame(metrics_data).T
    
    metrics_df.to_csv('model_evaluation_metrics.csv')
    print("Saved main metrics to model_evaluation_metrics.csv")
    
    # Print summary table for easy viewing in logs
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON - SUMMARY TABLE")
    print("="*70)
    print(metrics_df.to_string())
    print("="*70)
    
    # Print best model (F1-Score is primary metric: balances precision & recall)
    best_model = metrics_df['F1-Score'].idxmax()  # Find model with highest F1-Score
    best_f1 = metrics_df.loc[best_model, 'F1-Score']
    print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
    print("="*70)
    
    # Performance Analysis
    print("\n📊 PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Compare models
    print(f"\n1. Model Ranking by F1-Score (Primary Metric):")
    ranked = metrics_df.sort_values('F1-Score', ascending=False)
    for i, (model, row) in enumerate(ranked.iterrows(), 1):
        print(f"   {i}. {model}: {row['F1-Score']:.4f}")
    
    # Precision-Recall Trade-off Analysis
    print(f"\n2. Precision-Recall Trade-off Analysis:")
    for model in metrics_df.index:
        prec = metrics_df.loc[model, 'Precision']
        rec = metrics_df.loc[model, 'Recall']
        if prec > rec:
            print(f"   {model}: Higher Precision ({prec:.4f}) > Recall ({rec:.4f})")
            print(f"      → Better at avoiding false alarms, but misses more phishing sites")
        else:
            print(f"   {model}: Higher Recall ({rec:.4f}) > Precision ({prec:.4f})")
            print(f"      → Catches more phishing sites, but more false alarms")
    
    # Security Context Recommendation
    print(f"\n3. Security Context Recommendation:")
    best_recall_model = metrics_df['Recall'].idxmax()
    best_recall = metrics_df.loc[best_recall_model, 'Recall']
    print(f"   For phishing detection, RECALL is critical (catching malicious sites).")
    print(f"   {best_recall_model} has highest Recall ({best_recall:.4f}), detecting {best_recall*100:.2f}% of phishing sites.")
    print(f"   However, F1-Score balances both metrics - {best_model} is recommended overall.")
    
    # Performance Interpretation
    print(f"\n4. Overall Performance Interpretation:")
    avg_accuracy = metrics_df['Accuracy'].mean()
    if avg_accuracy > 0.85:
        print(f"   ✅ Excellent: Average accuracy {avg_accuracy:.4f} shows strong performance across all models.")
    elif avg_accuracy > 0.75:
        print(f"   ✓ Good: Average accuracy {avg_accuracy:.4f} indicates solid classification ability.")
    else:
        print(f"   ⚠ Fair: Average accuracy {avg_accuracy:.4f} suggests room for improvement.")
    
    print("="*70 + "\n")
