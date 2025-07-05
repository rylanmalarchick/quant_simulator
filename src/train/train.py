import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from src.ingest.sqlite_writer import get_db_connection
from src.config import load_config
import pickle
import os
from datetime import datetime
from src.logging import get_logger

logger = get_logger(__name__)

def train_model():
    """
    Trains the LightGBM model.
    """
    logger.info("Training model...")
    config = load_config()
    conn = get_db_connection()
    
    # Load features
    logger.info("Loading features...")
    features_df = pd.read_sql('SELECT * FROM features', conn)
    
    # Prepare data for training
    logger.info("Preparing data for training...")
    X = features_df.drop(['symbol', 'timestamp', 'close'], axis=1)
    y = (features_df['close'].shift(-1) > features_df['close']).astype(int) # Predict if next day's close is higher
    
    # Remove last row since we can't calculate target for it
    X = X[:-1]
    y = y[:-1]

    # 5-fold cross-validation
    logger.info("Performing 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    roc_aucs = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train model
        logger.info(f"Training fold {fold+1}...")
        model = lgb.LGBMClassifier(**config['model']['params'])
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Compute metrics
        accuracy = accuracy_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
        logger.info(f"Fold {fold+1} - Accuracy: {accuracy}, ROC-AUC: {roc_auc}")
        
    # Log metrics
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_roc_auc = sum(roc_aucs) / len(roc_aucs)
    logger.info(f"Average Accuracy: {avg_accuracy}")
    logger.info(f"Average ROC-AUC: {avg_roc_auc}")
    
    # Save model
    logger.info("Saving model...")
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, f"model_{datetime.now().strftime('%Y%m%d')}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    logger.info(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
