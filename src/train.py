from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.config import MODELS_TO_TRAIN, MODEL_PARAMS
import joblib
import os

def get_models():
    """ Defines the three models with configurations from config.py for easy experimentation. """
    # Define all available models with their hyperparameters (using ** to unpack dict)
    models = {
        'LogisticRegression': LogisticRegression(
            **MODEL_PARAMS['LogisticRegression']  # Unpack: solver='liblinear', class_weight='balanced', etc.
        ), # Selected for clear linear relationships
        'RandomForest': RandomForestClassifier(
            **MODEL_PARAMS['RandomForest']  # Unpack: n_estimators=100, n_jobs=-1, etc.
        ), # Selected to capture complex interactions
        'GradientBoosting': GradientBoostingClassifier(
            **MODEL_PARAMS['GradientBoosting']  # Unpack: n_estimators=100, learning_rate=0.1, etc.
        ) # Selected for sequential learning of hierarchical patterns
    }
    # Return only models specified in MODELS_TO_TRAIN (allows toggling models on/off)
    return {k: models[k] for k in MODELS_TO_TRAIN}

def train_models(X_train, y_train):
    """ Trains all models and saves them to disk. """
    models = get_models()
    trained_models = {}
    
    # Ensure models directory exists for saving artifacts
    os.makedirs('models', exist_ok=True)
    
    print("Starting model training...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Save the model artifact
        joblib.dump(model, f'models/{name}_model.pkl')
        print(f"Saved {name} to models/{name}_model.pkl")

    return trained_models
