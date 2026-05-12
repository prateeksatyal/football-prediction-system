"""
Generate predictions for upcoming matches with explanations.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from .models import ModelTrainer
from .feature_engineering import FeatureEngineer
from .database import db
from .config import IDX_TO_CLASS

logger = logging.getLogger(__name__)

class MatchPredictor:
    """Generate predictions for matches"""
    
    def __init__(self, model_trainer: ModelTrainer, feature_engineer: FeatureEngineer):
        self.trainer = model_trainer
        self.engineer = feature_engineer
    
    def predict_match(self, match_idx: int) -> Dict:
        """
        Predict outcome for a single match.
        
        Returns:
            {
                "match_id": int,
                "home_team": str,
                "away_team": str,
                "prediction": "H/D/A",
                "probabilities": {"H": float, "D": float, "A": float},
                "confidence": float,
                "key_drivers": List[Tuple[str, float]],
            }
        """
        match = self.engineer.matches_df.iloc[match_idx]
        
        # Compute features
        home_features, away_features = self.engineer.compute_all_features(match_idx)
        
        # Combine features
        features_combined = {
            **{f"home_{k}": v for k, v in home_features.items()},
            **{f"away_{k}": v for k, v in away_features.items()},
        }
        
        # Ensure all features are present
        for feature_name in self.trainer.feature_names:
            if feature_name not in features_combined:
                features_combined[feature_name] = 0
        
        X = np.array([features_combined[f] for f in self.trainer.feature_names]).reshape(1, -1)
        
        # Predict
        proba = self.trainer.ensemble_predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_result = IDX_TO_CLASS[pred_idx]
        confidence = float(proba[pred_idx])
        
        return {
            "match_id": match["match_id"],
            "home_team_id": match["home_team_id"],
            "away_team_id": match["away_team_id"],
            "date": match["date"],
            "prediction": pred_result,
            "prob_home": float(proba[0]),
            "prob_draw": float(proba[1]),
            "prob_away": float(proba[2]),
            "confidence": confidence,
            "key_drivers": self._extract_key_drivers(features_combined),
        }
    
    def _extract_key_drivers(self, features: Dict, top_n: int = 5) -> List[Tuple[str, float]]:
        """Extract top features driving prediction"""
        # Simple heuristic: use features with highest absolute values
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:top_n]
    
    def predict_upcoming_matches(self, n_matches: int = 10) -> List[Dict]:
        """Predict all upcoming matches"""
        logger.info(f"Generating predictions for upcoming matches...")
        
        upcoming_matches = db.get_upcoming_matches()
        
        if len(upcoming_matches) == 0:
            logger.warning("No upcoming matches found")
            return []
        
        predictions = []
        for _, match in upcoming_matches.head(n_matches).iterrows():
            # Find match index in engineer's matches_df
            match_idx = self.engineer.matches_df[
                self.engineer.matches_df["match_id"] == match["match_id"]
            ].index
            
            if len(match_idx) > 0:
                pred = self.predict_match(match_idx[0])
                predictions.append(pred)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions