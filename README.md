Phishing Website Detection ML Pipeline

##  Problem Statement

As a new hire to the ML Engineering team at CyberProtect, you have been entrusted with a critical task: protecting the users of CyberProtect from phishing attacks when they are assessing websites on the internet.
CyberProtect has collected a database of phishing websites and legitimate websites. 

Task: Develop predictive models that can be installed as an extension and predict if a website is a phishing attack and warn the user before allowing the user to access it. Build and evaluate prediction models, and also identify their respective key features of the dataset that categorise whether the website is a phishing attack. 
---

## 📁 Project Structure

```
aiap22-ng-jia-li-138H/
├── data/
│   └── phishing.db              # SQLite database (downloaded by run.sh)
├── src/
│   ├── config.py                # Configuration and constants
│   ├── data_loader.py           # Database connection and data loading
│   ├── preprocessing.py         # Data cleaning and feature engineering
│   ├── train.py                 # Model training logic
│   ├── evaluate.py              # Model evaluation and metrics
│   └── main_pipeline.py         # Main pipeline orchestrator
├── .github/
│   └── workflows/               # GitHub Actions CI/CD workflows
├── eda.ipynb                    # Exploratory Data Analysis notebook
├── requirements.txt             # Python dependencies
├── run.sh                       # Pipeline execution script
└── README.md                    # This file
```

---

## 🚀 Execution Instructions

### Prerequisites
- Python 3.9 or higher
- Internet connection (to download database)

### Running the Pipeline

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute the pipeline:**
   ```bash
   bash run.sh
   ```

   This script will:
   - Download the phishing database from the remote source
   - Execute the complete ML pipeline (preprocessing, training, evaluation)
   - Save trained models to `models/` directory
   - Generate `model_evaluation_metrics.csv` with performance metrics

### Pipeline Outputs

After successful execution, the following artifacts are generated:

**1. Model Files (`models/` directory):**
- `RandomForest_model.pkl` (~26 MB) - Ensemble model with 100 trees
- `GradientBoosting_model.pkl` (~185 KB) - Sequential boosting model
- `LogisticRegression_model.pkl` (~2.3 KB) - Linear baseline model
- `scaler.pkl` (~1.2 KB) - StandardScaler for feature normalization

**2. Evaluation Results:**
- `model_evaluation_metrics.csv` - Performance comparison table with 4 metrics per model
  - Format: `Model, Accuracy, Precision, Recall, F1-Score`
  - Best model highlighted in console output with 🏆 emoji

**3. Console Output:**
- Data cleaning progress (✅ checkmarks for each step)
- Model training status (per-model training confirmation)
- Detailed evaluation metrics (confusion matrices, metric breakdowns)
- Performance analysis (model ranking, precision-recall trade-offs, recommendations)

**Note**: Model `.pkl` files are excluded from git (see `.gitignore`) as they are regenerated on each pipeline run. To use saved models:
```python
import joblib
model = joblib.load('models/GradientBoosting_model.pkl')
scaler = joblib.load('models/scaler.pkl')
```

### Viewing ML Pipeline Results

**For Assessors - GitHub Actions Workflow Logs:**

When the pipeline runs via GitHub Actions, all outputs are visible in the workflow logs:

1. **Navigate to**: Actions tab → Select latest workflow run
2. **Expand**: "Run executable bash script" step
3. **View complete output including**:
   - Data preprocessing steps with ✅ confirmations
   - Model training progress for all 3 models
   - **Confusion matrices** for each model (2x2 matrix showing TP/TN/FP/FN)
   - **Performance metrics table**: Accuracy, Precision, Recall, F1-Score for all models
   - **🏆 Best Model identification**: Winner by F1-Score
   - **📊 Performance Analysis**:
     - Model ranking (1st, 2nd, 3rd by F1-Score)
     - Precision-Recall trade-off analysis per model
     - Security context recommendations (why Recall matters for phishing)
     - Overall performance interpretation (excellent/good/fair assessment)

**Example Log Output:**
```
======================================================================
FINAL MODEL COMPARISON - SUMMARY TABLE
======================================================================
                    Accuracy  Precision    Recall  F1-Score
GradientBoosting    0.8371    0.8680      0.8304    0.8488
RandomForest        0.8229    0.8305      0.8521    0.8412
LogisticRegression  0.7824    0.7920      0.8201    0.8058

🏆 Best Model: GradientBoosting (F1-Score: 0.8488)

📊 PERFORMANCE ANALYSIS
1. Model Ranking by F1-Score (Primary Metric):
   1. GradientBoosting: 0.8488
   2. RandomForest: 0.8412
   3. LogisticRegression: 0.8058

2. Precision-Recall Trade-off Analysis:
   GradientBoosting: Higher Precision (0.8680) > Recall (0.8304)
      → Better at avoiding false alarms, but misses more phishing sites
   ...
```

**Local Execution:**
- Run `bash run.sh` to see identical output in terminal
- `model_evaluation_metrics.csv` saved to project root
- Model `.pkl` files saved to `models/` directory

### Modifying Parameters

**Model Selection** (`src/config.py`):
```python
MODELS_TO_TRAIN = ['RandomForest', 'LogisticRegression', 'GradientBoosting']
```

**Feature Configuration** (`src/config.py`):
```python
NUMERICAL_COLS = [...]           # Features to standardize
BINARY_INT_COLS = [...]          # Binary features
CATEGORICAL_NOMINAL_COLS = [...] # Features for one-hot encoding
```

**Train-Test Split** (`src/preprocessing.py`):
```python
def split_data(X, y, test_size=0.2, random_state=42):  # Modify test_size here
```

---

## 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING (data_loader.py)                           │
│    - Connect to SQLite database                             │
│    - Load raw phishing data                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. DATA CLEANING (preprocessing.py - clean_data)           │
│    - Drop redundant columns (Unnamed: 0)                    │
│    - Impute missing values (LineOfCode median)              │
│    - Correct invalid values (negative NoOfImage → 0)        │
│    - Standardize categorical text (lowercase, strip)        │
│    - Convert binary features to int                         │
│    - Cap outliers using IQR method                          │
│    - **Detect & fix data contamination**                    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING (preprocessing.py)                  │
│    - One-hot encode categorical features                    │
│    - Standardize numerical features (StandardScaler)        │
│    - Create feature matrix X and target vector y           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 4. DATA SPLITTING (preprocessing.py)                       │
│    - 80% training, 20% testing                              │
│    - Stratified split (preserve class distribution)         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 5. MODEL TRAINING (train.py)                               │
│    - Train RandomForest, LogisticRegression, GradientBoost  │
│    - Save model artifacts to models/                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 6. MODEL EVALUATION (evaluate.py)                          │
│    - Calculate metrics: Accuracy, Precision, Recall, F1     │
│    - Generate confusion matrices                            │
│    - Log results to CSV and console                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Key EDA Findings & Pipeline Decisions

### Data Quality Issues Identified
1. **Missing Values**: 22.43% missing in `LineOfCode` → **Imputed with median (620.0)**
2. **Invalid Values**: 377 negative values in `NoOfImage` → **Corrected to 0**
3. **Redundant Column**: `Unnamed: 0` (auto-generated index) → **Dropped**
4. **Categorical Inconsistencies**: Mixed case and whitespace in `Industry` → **Standardized (lowercase, stripped)**
5. **Outliers**: Extreme values in 7 numerical features → **Capped using IQR method**
6. **Data Contamination**: Non-numeric values (e.g., `'000webhost'`) detected in numeric columns (Section 2.7) → **Detected and replaced with median/mode (Section 3.5)**

### Feature Engineering Decisions
- **One-Hot Encoding**: Applied to `Industry` (nominal categorical)
- **Standardization**: Applied to all 9 numerical features using `StandardScaler` for consistent scale
- **Binary Features**: `Robots` and `IsResponsive` converted to int (0/1) for model compatibility

---

## 📊 Feature Processing Summary

| Feature               | Type        | Processing Applied                          | Rationale                                      |
|-----------------------|-------------|---------------------------------------------|------------------------------------------------|
| **LineOfCode**        | Numerical   | Median imputation (620.0), Standardization  | Handle 22% missing values, normalize scale     |
| **LargestLineLength** | Numerical   | IQR outlier capping, Standardization        | Reduce extreme value influence                 |
| **NoOfURLRedirect**   | Numerical   | IQR outlier capping, Standardization        | Reduce extreme value influence                 |
| **NoOfPopup**         | Numerical   | IQR outlier capping, Standardization        | Reduce extreme value influence                 |
| **NoOfiFrame**        | Numerical   | IQR outlier capping, Standardization        | Reduce extreme value influence                 |
| **NoOfImage**         | Numerical   | Negative→0, IQR capping, Standardization    | Fix invalid values, reduce outliers            |
| **NoOfSelfRef**       | Numerical   | IQR outlier capping, Standardization        | Reduce extreme value influence                 |
| **NoOfExternalRef**   | Numerical   | IQR outlier capping, Standardization        | Reduce extreme value influence                 |
| **DomainAgeMonths**   | Numerical   | Standardization only                        | Normalize scale                                |
| **Robots**            | Binary      | Convert to int (0/1)                        | Ensure numeric type for modeling               |
| **IsResponsive**      | Binary      | Convert to int (0/1)                        | Ensure numeric type for modeling               |
| **Industry**          | Categorical | Lowercase, strip whitespace, One-hot encode | Standardize text, convert to numeric features  |
| **label**             | Target      | No processing (used as-is)                  | Binary classification target                   |

---

## 🤖 Model Selection Rationale

### EDA Findings → Model Choices

EDA insights directly informed our model selection:

- **Linear relationships observed** (correlations 0.33-0.39) → **Logistic Regression** as baseline
- **Non-linear patterns in KDE plots** (distinct distributions, minimal overlap) → **Random Forest** to capture feature interactions
- **6 features with 200-300% mean differences** (NoOfSelfRef, NoOfExternalRef, NoOfImage, LineOfCode) → **Gradient Boosting** for hierarchical pattern learning
- **Mild class imbalance** (55%/45% split) → `class_weight='balanced'` applied to all models
- **No multicollinearity** (all feature-to-feature correlations < 0.7) → retain all features without elimination

### Models Chosen

| Model                   | Rationale                                                                                     |
|-------------------------|-----------------------------------------------------------------------------------------------|
| **Logistic Regression** | - Baseline linear model for binary classification<br>- Interpretable coefficients<br>- Fast training<br>- `class_weight='balanced'` to handle mild class imbalance |
| **Random Forest**       | - Captures non-linear relationships and feature interactions<br>- Robust to outliers<br>- Feature importance insights<br>- `class_weight='balanced'` for imbalance handling |
| **Gradient Boosting**   | - Sequential learning of patterns<br>- Often achieves higher accuracy<br>- Captures complex hierarchical relationships<br>- Strong performance on tabular data |

### Hyperparameter Selection

**Approach**: Centralized configuration in `src/config.py` for easy experimentation

All model hyperparameters, data split ratios, and preprocessing settings are defined in `config.py`:

```python
# Easy experimentation - change parameters here!
MODEL_PARAMS = {
    'LogisticRegression': {
        'solver': 'liblinear',
        'class_weight': 'balanced',
        'max_iter': 1000,
        'random_state': 42
    },
    'RandomForest': {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    },
    'GradientBoosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Train-test split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Outlier capping columns
OUTLIER_COLS = ['LargestLineLength', 'NoOfURLRedirect', ...]
```

**Benefits**:
- ✅ **Single source of truth**: All parameters in one location
- ✅ **Easy experimentation**: Change `n_estimators=200`, re-run pipeline immediately
- ✅ **No code changes**: Modify `config.py` only, never touch training logic
- ✅ **Reproducibility**: Random states centralized for consistent results

**Current Configuration Rationale**: 
- Default hyperparameters chosen based on EDA findings (clear class separation with 200-300% feature differences)
- Production deployment would benefit from grid/random search optimization using these as baseline

---

## 📈 Model Evaluation

### Metrics Used

| Metric        | Description                                                                 | Why Important                                      |
|---------------|-----------------------------------------------------------------------------|----------------------------------------------------|
| **Accuracy**  | Proportion of correct predictions (TP+TN)/(TP+TN+FP+FN)                     | Overall model correctness                          |
| **Precision** | Proportion of true positives among predicted positives (TP/(TP+FP))         | Minimize false alarms (legitimate sites flagged)   |
| **Recall**    | Proportion of actual positives correctly identified (TP/(TP+FN))            | Minimize missed phishing sites (critical for security) |
| **F1-Score**  | Harmonic mean of Precision and Recall (2×P×R/(P+R))                         | Balance between Precision and Recall               |

### Validation Strategy

- **Train-Test Split**: 80% training, 20% testing (stratified to preserve 55%/45% class distribution)
- **Random State**: Fixed seed (42) ensures reproducibility across runs
- **No Cross-Validation**: Single hold-out test set used due to large dataset size (10,500 rows)
- **Stratification**: Applied to maintain class balance in both train/test sets

### Model Selection Process

1. **Train all 3 models** on identical training data (80% of 10,500 = 8,400 samples)
2. **Evaluate on held-out test set** (20% = 2,100 samples) - never seen during training
3. **Calculate 4 metrics** for each model: Accuracy, Precision, Recall, F1-Score
4. **Compare F1-Scores** as primary selection criterion (balances precision/recall)
5. **Select best model** based on highest F1-Score
6. **Log results** to `model_evaluation_metrics.csv` for reproducible comparison

### Performance Results

*(Results will be displayed in console output and saved to `model_evaluation_metrics.csv`)*

**Best Model Selection Criteria:**
- Prioritize **F1-Score** for balanced precision-recall tradeoff
- **Recall** is critical: missing a phishing site is costly (security risk)
- **Precision** prevents user friction from false positives (legitimate sites flagged)

---

## 🚢 Deployment Considerations

### Model Deployment
1. **Model Persistence**: Models saved as `.pkl` files using `joblib` for easy loading
2. **Preprocessing Pipeline**: All preprocessing steps must be applied to new data in the same order
3. **Feature Consistency**: Ensure incoming data has same features and encoding as training data

### Production Requirements
- **Real-time Inference**: Logistic Regression offers fastest predictions (~ms latency)
- **Batch Processing**: Random Forest/Gradient Boosting acceptable for offline scoring
- **Model Monitoring**: Track prediction distribution, feature drift, and performance degradation
- **Retraining Schedule**: Retrain quarterly as phishing tactics evolve
- **Preprocessing Artifacts**: Scaler saved to `models/scaler.pkl` for consistent feature transformation in production

### Scalability
- **Horizontal Scaling**: Stateless prediction API can scale with containerization (Docker/Kubernetes)
- **Model Registry**: Version control models with MLflow or similar tools
- **A/B Testing**: Deploy new models alongside existing ones to validate improvements

### Security & Privacy
- **Input Validation**: Sanitize URLs and features before prediction
- **Rate Limiting**: Prevent API abuse with request throttling
- **Data Retention**: Comply with data protection regulations (GDPR, PDPA)
- **Model Explainability**: Use SHAP/LIME for transparency in high-stakes decisions

---

## 📝 Notes

- **GitHub Actions Compatibility**: Pipeline automatically runs on push via `.github/workflows`
  - **Outputs Visible in Logs**: All model evaluation results printed to console (accessible in Actions workflow logs)
  - **CSV Generated Locally**: `model_evaluation_metrics.csv` created during pipeline execution (view in logs)
  - **Performance Analysis**: Detailed metric interpretation, model ranking, and recommendations shown in workflow output
- **Database Source**: Automatically downloaded from Azure Blob Storage by `run.sh`
- **Reproducibility**: Fixed random seeds (42) ensure consistent reproducible results across runs (model trained with same train/test split)
- **Minimal Dependencies**: Only essential packages in `requirements.txt` for lean deployment

---

