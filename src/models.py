"""
Model training, ensemble, and prediction.
"""
import pickle
import logging
from typing import Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb

from .config import (
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and manage ensemble of models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_names = None
    
    def _get_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels"""
        feature_cols = [c for c in df.columns if c not in ["match_id", "date", "label", 
                                                            "home_team_id", "away_team_id"]]
        X = df[feature_cols].fillna(0).values
        y = df["label"].map(CLASS_TO_IDX).values
        
        self.feature_names = feature_cols
        return X, y
    
    def train_baseline_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train baseline logistic regression"""
        logger.info("Training baseline logistic regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        
        model.fit(X, y)
        self.models["logistic_regression"] = model
        
        train_acc = model.score(X, y)
        logger.info(f"Logistic Regression training accuracy: {train_acc:.4f}")
        
        return {"accuracy": train_acc, "model": model}
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train XGBoost classifier"""
        logger.info("Training XGBoost...")
        
        model = xgb.XGBClassifier(
            **XGBOOST_PARAMS,
            scale_pos_weight=1,
        )
        
        model.fit(X, y, verbose=False)
        self.models["xgboost"] = model
        
        train_acc = model.score(X, y)
        logger.info(f"XGBoost training accuracy: {train_acc:.4f}")
        
        return {"accuracy": train_acc, "model": model}
    
    def train_lightgbm(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train LightGBM classifier"""
        logger.info("Training LightGBM...")
        
        model = lgb.LGBMClassifier(
            **LIGHTGBM_PARAMS,
        )
        
        model.fit(X, y, verbose=0)
        self.models["lightgbm"] = model
        
        train_acc = model.score(X, y)
        logger.info(f"LightGBM training accuracy: {train_acc:.4f}")
        
        return {"accuracy": train_acc, "model": model}
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Train all models and evaluate"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING ENSEMBLE MODELS")
        logger.info("="*60)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train baseline
        self.train_baseline_logistic_regression(X_train_scaled, y_train)
        lr_preds = self.models["logistic_regression"].predict(X_test_scaled)
        results["logistic_regression"] = {
            "accuracy": accuracy_score(y_test, lr_preds),
            "precision": precision_score(y_test, lr_preds, average="weighted", zero_division=0),
            "recall": recall_score(y_test, lr_preds, average="weighted", zero_division=0),
        }
        
        # Train XGBoost
        self.train_xgboost(X_train, y_train)
        xgb_preds = self.models["xgboost"].predict(X_test)
        results["xgboost"] = {
            "accuracy": accuracy_score(y_test, xgb_preds),
            "precision": precision_score(y_test, xgb_preds, average="weighted", zero_division=0),
            "recall": recall_score(y_test, xgb_preds, average="weighted", zero_division=0),
        }
        
        # Train LightGBM
        self.train_lightgbm(X_train, y_train)
        lgb_preds = self.models["lightgbm"].predict(X_test)
        results["lightgbm"] = {
            "accuracy": accuracy_score(y_test, lgb_preds),
            "precision": precision_score(y_test, lgb_preds, average="weighted", zero_division=0),
            "recall": recall_score(y_test, lgb_preds, average="weighted", zero_division=0),
        }
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
        
        return results
    
    def ensemble_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions with probability averaging"""
        proba_list = []
        
        for model_name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                proba_list.append(proba)
        
        # Average probabilities
        ensemble_proba = np.mean(proba_list, axis=0)
        return ensemble_proba
    
    def ensemble_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions"""
        proba = self.ensemble_predict_proba(X)
        preds = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)
        
        return preds, confidence
    
    def save_models(self, path: str):
        """Save all trained models"""
        logger.info(f"Saving models to {path}")
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            }, f)
    
    def load_models(self, path: str):
        """Load trained models"""
        logger.info(f"Loading models from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.models = data["models"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]

def time_series_cross_validation(features_df: pd.DataFrame, split_date: str = None, n_splits: int = 3):
    """
    Perform time-series cross-validation.
    
    For sports, we use forward-chaining: train on past, test on future.
    """
    logger.info(f"Performing time-series cross-validation ({n_splits} splits)...")
    
    features_df = features_df.sort_values("date").reset_index(drop=True)
    
    n = len(features_df)
    fold_size = n // (n_splits + 1)
    
    fold_accuracies = []
    
    for fold in range(n_splits):
        train_end = (fold + 1) * fold_size
        test_end = (fold + 2) * fold_size
        
        train_df = features_df.iloc[:train_end]
        test_df = features_df.iloc[train_end:test_end]
        
        if len(test_df) == 0:
            continue
        
        trainer = ModelTrainer()
        X_train, y_train = trainer._get_xy(train_df)
        X_test, y_test = trainer._get_xy(test_df)
        
        trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        preds, _ = trainer.ensemble_predict(X_test)
        acc = accuracy_score(y_test, preds)
        fold_accuracies.append(acc)
        
        logger.info(f"Fold {fold + 1} accuracy: {acc:.4f}")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    logger.info(f"\nTime-series CV Results:")
    logger.info(f"  Mean Accuracy: {mean_acc:.4f}")
    logger.info(f"  Std Dev:       {std_acc:.4f}")
    
    return mean_acc, std_acc