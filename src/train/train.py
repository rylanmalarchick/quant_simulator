import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from src.ingest.sqlite_writer import get_db_connection
from src.config import load_config
import pickle
import os
from datetime import datetime
from src.logging import get_logger

logger = get_logger(__name__)

# Ensemble configuration
ENSEMBLE_SEEDS = [42, 123, 456]  # 3 different seeds for model diversity


class EnsembleModel:
    """
    Ensemble of LightGBM models with different random seeds.
    Averages predictions for more robust signals.
    """
    def __init__(self, models):
        self.models = models
        self.feature_name_ = models[0].feature_name_
    
    def predict_proba(self, X):
        """Average probability predictions across all models."""
        probas = np.array([m.predict_proba(X) for m in self.models])
        return probas.mean(axis=0)
    
    def predict(self, X):
        """Predict class based on averaged probabilities."""
        avg_proba = self.predict_proba(X)
        return (avg_proba[:, 1] > 0.5).astype(int)


def train_model():
    """
    Trains an ensemble of LightGBM models using TimeSeriesSplit.
    
    IMPORTANT: We use TimeSeriesSplit instead of KFold because:
    - Financial data is time-ordered; future data should never leak into training
    - KFold with shuffle=True randomly mixes future and past data, causing overfitting
    - TimeSeriesSplit ensures we only train on past data and validate on future data
    
    ENSEMBLE: We train 3 models with different random seeds and average their
    predictions. This reduces variance and produces more stable signals.
    """
    logger.info("Training ensemble model (3 seeds for robustness)...")
    config = load_config()
    conn = get_db_connection()
    
    # Load features
    logger.info("Loading features...")
    features_df = pd.read_sql('SELECT * FROM features', conn)
    
    # Sort by timestamp to ensure proper time ordering
    features_df = features_df.sort_values('timestamp').reset_index(drop=True)
    
    # Prepare data for training
    logger.info("Preparing data for training...")
    X = features_df.drop(['symbol', 'timestamp', 'close'], axis=1)
    y = (features_df['close'].shift(-1) > features_df['close']).astype(int)  # Predict if next day's close is higher
    
    # Remove last row since we can't calculate target for it
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    # Handle any NaN values that may exist in features
    X = X.fillna(0)

    # TimeSeriesSplit cross-validation (prevents data leakage)
    # Uses expanding window: train on [0:i], validate on [i:i+1]
    logger.info("Performing TimeSeriesSplit cross-validation (5 splits)...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Track metrics for each seed
    all_accuracies = {seed: [] for seed in ENSEMBLE_SEEDS}
    all_roc_aucs = {seed: [] for seed in ENSEMBLE_SEEDS}
    ensemble_accuracies = []
    ensemble_roc_aucs = []
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        logger.info(f"Training fold {fold+1} (train: {len(train_index)}, val: {len(val_index)})...")
        
        fold_models = []
        fold_probas = []
        
        for seed in ENSEMBLE_SEEDS:
            # Get params and override seed
            params = config['model']['params'].copy()
            params['random_state'] = seed
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            fold_models.append(model)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            fold_probas.append(y_pred_proba)
            
            # Individual model metrics
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            try:
                roc_auc = roc_auc_score(y_val, y_pred_proba)
            except ValueError:
                roc_auc = 0.5
            all_accuracies[seed].append(accuracy)
            all_roc_aucs[seed].append(roc_auc)
        
        # Ensemble predictions (average probabilities)
        ensemble_proba = np.mean(fold_probas, axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        ensemble_acc = accuracy_score(y_val, ensemble_pred)
        try:
            ensemble_auc = roc_auc_score(y_val, ensemble_proba)
        except ValueError:
            ensemble_auc = 0.5
        
        ensemble_accuracies.append(ensemble_acc)
        ensemble_roc_aucs.append(ensemble_auc)
        logger.info(f"Fold {fold+1} - Ensemble Accuracy: {ensemble_acc:.4f}, ROC-AUC: {ensemble_auc:.4f}")
        
    # Log comparison
    logger.info("=== Model Comparison ===")
    for seed in ENSEMBLE_SEEDS:
        avg_acc = np.mean(all_accuracies[seed])
        avg_auc = np.mean(all_roc_aucs[seed])
        logger.info(f"  Seed {seed}: Accuracy={avg_acc:.4f}, ROC-AUC={avg_auc:.4f}")
    
    avg_ensemble_acc = np.mean(ensemble_accuracies)
    avg_ensemble_auc = np.mean(ensemble_roc_aucs)
    logger.info(f"  ENSEMBLE: Accuracy={avg_ensemble_acc:.4f}, ROC-AUC={avg_ensemble_auc:.4f}")
    
    # Train final ensemble on all data
    logger.info("Training final ensemble on all data...")
    final_models = []
    for seed in ENSEMBLE_SEEDS:
        params = config['model']['params'].copy()
        params['random_state'] = seed
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        final_models.append(model)
    
    ensemble = EnsembleModel(final_models)
    
    # Save ensemble model
    logger.info("Saving ensemble model...")
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, f"model_{datetime.now().strftime('%Y%m%d')}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
        
    logger.info(f"Ensemble model saved to {model_path}")
    
    return {
        'avg_accuracy': avg_ensemble_acc,
        'avg_roc_auc': avg_ensemble_auc,
        'model_path': model_path
    }

if __name__ == '__main__':
    train_model()
