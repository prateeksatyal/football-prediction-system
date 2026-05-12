"""
Advanced feature engineering for football match prediction.
Creates 100+ features from historical match data.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import dict, Tuple

from .config import ROLLING_WINDOWS, MIN_MATCHES_REQUIRED
from .utils import days_between, get_utc_now

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for football matches.
    
    Feature categories:
    1. Rolling form statistics (3, 5, 10, 15, 20 matches)
    2. Head-to-head history
    3. Home/away splits
    4. Fixture difficulty
    5. Elo ratings
    6. Rest days and fixture congestion
    7. League position
    """
    
    def __init__(self, matches_df: pd.DataFrame):
        """
        Initialize with all historical matches.
        
        Args:
            matches_df: DataFrame with columns:
                - match_id, date, home_team_id, away_team_id, league, season
                - home_goals, away_goals, result (H/D/A)
        """
        self.matches_df = matches_df.sort_values("date").reset_index(drop=True)
        self.elo_ratings = {}  # team_id -> current Elo rating
        self._initialize_elo_ratings()
    
    def _initialize_elo_ratings(self, initial_rating: float = 1500):
        """Initialize Elo ratings for all teams"""
        unique_teams = set(
            list(self.matches_df["home_team_id"].unique()) + 
            list(self.matches_df["away_team_id"].unique())
        )
        for team_id in unique_teams:
            if pd.notna(team_id):
                self.elo_ratings[int(team_id)] = initial_rating
    
    def _calculate_elo_change(self, expected_score: float, actual_score: float, 
                              k_factor: float = 32) -> float:
        """Calculate Elo rating change"""
        return k_factor * (actual_score - expected_score)
    
    def _expected_score_elo(self, rating1: float, rating2: float) -> float:
        """Calculate expected score in Elo system"""
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    
    def compute_all_features(self, match_idx: int) -> Tuple[dict, dict]:
        """
        Compute features for a match (at match index in sorted matches_df).
        
        Returns:
            (home_features, away_features)
            Each as dict with 100+ features
        """
        match = self.matches_df.iloc[match_idx]
        
        # Get all matches BEFORE this one
        prev_matches = self.matches_df.iloc[:match_idx]
        
        home_team_id = match["home_team_id"]
        away_team_id = match["away_team_id"]
        match_date = match["date"]
        league = match["league"]
        
        # Compute features
        home_features = self._compute_team_features(
            home_team_id, away_team_id, match_date, prev_matches, 
            league, is_home=True
        )
        
        away_features = self._compute_team_features(
            away_team_id, home_team_id, match_date, prev_matches,
            league, is_home=False
        )
        
        return home_features, away_features
    
    def _compute_team_features(self, team_id: int, opponent_id: int, 
                              match_date: datetime, prev_matches: pd.DataFrame,
                              league: str, is_home: bool) -> dict:
        """Compute features for one team in a match"""
        
        features = {}
        
        # Get team's previous matches (home or away)
        if is_home:
            team_matches = prev_matches[prev_matches["home_team_id"] == team_id].copy()
            opponent_matches = prev_matches[prev_matches["away_team_id"] == team_id].copy()
        else:
            team_matches = prev_matches[prev_matches["away_team_id"] == team_id].copy()
            opponent_matches = prev_matches[prev_matches["home_team_id"] == team_id].copy()
        
        all_team_matches = pd.concat([team_matches, opponent_matches]).sort_values("date")
        
        # 1. ROLLING FORM FEATURES
        for window in ROLLING_WINDOWS:
            suffix = f"_last_{window}"
            recent_matches = team_matches.tail(window)
            
            if len(recent_matches) < MIN_MATCHES_REQUIRED:
                features[f"form_wins{suffix}"] = 0
                features[f"form_draws{suffix}"] = 0
                features[f"form_losses{suffix}"] = 0
                features[f"form_gf{suffix}"] = 0
                features[f"form_ga{suffix}"] = 0
                features[f"form_gd{suffix}"] = 0
                features[f"form_pts{suffix}"] = 0
                features[f"form_win_rate{suffix}"] = 0.0
                continue
            
            if is_home:
                results = recent_matches["result"].value_counts()
                gf = recent_matches["home_goals"].sum()
                ga = recent_matches["away_goals"].sum()
            else:
                results = recent_matches["result"].map({"H": "A", "D": "D", "A": "H"}).value_counts()
                gf = recent_matches["away_goals"].sum()
                ga = recent_matches["home_goals"].sum()
            
            wins = results.get("H" if is_home else "A", 0)
            draws = results.get("D", 0)
            losses = results.get("A" if is_home else "H", 0)
            
            features[f"form_wins{suffix}"] = wins
            features[f"form_draws{suffix}"] = draws
            features[f"form_losses{suffix}"] = losses
            features[f"form_gf{suffix}"] = gf
            features[f"form_ga{suffix}"] = ga
            features[f"form_gd{suffix}"] = gf - ga
            features[f"form_pts{suffix}"] = wins * 3 + draws
            features[f"form_win_rate{suffix}"] = wins / len(recent_matches) if len(recent_matches) > 0 else 0
        
        # 2. HEAD-TO-HEAD FEATURES
        h2h = all_team_matches[
            ((all_team_matches["home_team_id"] == team_id) & 
             (all_team_matches["away_team_id"] == opponent_id)) |
            ((all_team_matches["away_team_id"] == team_id) & 
             (all_team_matches["home_team_id"] == opponent_id))
        ]
        
        if len(h2h) > 0:
            if is_home:
                h2h_home = h2h[h2h["home_team_id"] == team_id]
                h2h_wins = (h2h_home["result"] == "H").sum()
                h2h_draws = (h2h_home["result"] == "D").sum()
                h2h_losses = (h2h_home["result"] == "A").sum()
                h2h_gf = h2h_home["home_goals"].sum()
                h2h_ga = h2h_home["away_goals"].sum()
            else:
                h2h_away = h2h[h2h["away_team_id"] == team_id]
                h2h_wins = (h2h_away["result"] == "A").sum()
                h2h_draws = (h2h_away["result"] == "D").sum()
                h2h_losses = (h2h_away["result"] == "H").sum()
                h2h_gf = h2h_away["away_goals"].sum()
                h2h_ga = h2h_away["home_goals"].sum()
            
            features["h2h_matches"] = len(h2h)
            features["h2h_wins"] = h2h_wins
            features["h2h_draws"] = h2h_draws
            features["h2h_losses"] = h2h_losses
            features["h2h_gf"] = h2h_gf
            features["h2h_ga"] = h2h_ga
            features["h2h_win_pct"] = h2h_wins / len(h2h) if len(h2h) > 0 else 0
        else:
            features["h2h_matches"] = 0
            features["h2h_wins"] = 0
            features["h2h_draws"] = 0
            features["h2h_losses"] = 0
            features["h2h_gf"] = 0
            features["h2h_ga"] = 0
            features["h2h_win_pct"] = 0.0
        
        # 3. HOME/AWAY SPLIT FEATURES
        home_record = team_matches[team_matches["home_team_id"] == team_id]
        away_record = team_matches[team_matches["away_team_id"] == team_id]
        
        if is_home and len(home_record) > 0:
            features["home_wins"] = (home_record["result"] == "H").sum()
            features["home_gf"] = home_record["home_goals"].sum()
            features["home_ga"] = home_record["away_goals"].sum()
            features["home_gd"] = features["home_gf"] - features["home_ga"]
            features["home_pts"] = features["home_wins"] * 3 + (home_record["result"] == "D").sum()
            features["home_avg_gf"] = features["home_gf"] / len(home_record)
            features["home_avg_ga"] = features["home_ga"] / len(home_record)
        else:
            features["home_wins"] = 0
            features["home_gf"] = 0
            features["home_ga"] = 0
            features["home_gd"] = 0
            features["home_pts"] = 0
            features["home_avg_gf"] = 0
            features["home_avg_ga"] = 0
        
        if not is_home and len(away_record) > 0:
            features["away_wins"] = (away_record["result"] == "A").sum()
            features["away_gf"] = away_record["away_goals"].sum()
            features["away_ga"] = away_record["home_goals"].sum()
            features["away_gd"] = features["away_gf"] - features["away_ga"]
            features["away_pts"] = features["away_wins"] * 3 + (away_record["result"] == "D").sum()
            features["away_avg_gf"] = features["away_gf"] / len(away_record)
            features["away_avg_ga"] = features["away_ga"] / len(away_record)
        else:
            features["away_wins"] = 0
            features["away_gf"] = 0
            features["away_ga"] = 0
            features["away_gd"] = 0
            features["away_pts"] = 0
            features["away_avg_gf"] = 0
            features["away_avg_ga"] = 0
        
        # 4. RECENT REST DAYS
        if len(team_matches) > 0:
            last_match_date = team_matches.iloc[-1]["date"]
            rest_days = days_between(last_match_date, match_date)
            features["days_rest"] = rest_days
            features["days_rest_sq"] = rest_days ** 2
        else:
            features["days_rest"] = 0
            features["days_rest_sq"] = 0
        
        # 5. OVERALL STATS
        if len(all_team_matches) > 0:
            features["total_matches_played"] = len(all_team_matches)
            if is_home:
                gf_total = all_team_matches[all_team_matches["home_team_id"] == team_id]["home_goals"].sum()
                ga_total = all_team_matches[all_team_matches["home_team_id"] == team_id]["away_goals"].sum()
            else:
                gf_total = all_team_matches[all_team_matches["away_team_id"] == team_id]["away_goals"].sum()
                ga_total = all_team_matches[all_team_matches["away_team_id"] == team_id]["home_goals"].sum()
            
            features["career_gf"] = gf_total
            features["career_ga"] = ga_total
            features["career_gd"] = gf_total - ga_total
            features["career_avg_gf"] = gf_total / len(all_team_matches)
            features["career_avg_ga"] = ga_total / len(all_team_matches)
        else:
            features["total_matches_played"] = 0
            features["career_gf"] = 0
            features["career_ga"] = 0
            features["career_gd"] = 0
            features["career_avg_gf"] = 0
            features["career_avg_ga"] = 0
        
        # 6. OPPONENT STRENGTH
        opponent_matches = all_team_matches[
            (all_team_matches["home_team_id"] == opponent_id) | 
            (all_team_matches["away_team_id"] == opponent_id)
        ]
        
        if len(opponent_matches) > 0:
            features["opponent_total_matches"] = len(opponent_matches)
            features["opponent_avg_gf"] = opponent_matches.apply(
                lambda x: x["home_goals"] if x["home_team_id"] == opponent_id else x["away_goals"],
                axis=1
            ).mean()
        else:
            features["opponent_total_matches"] = 0
            features["opponent_avg_gf"] = 0
        
        # 7. ELO RATING
        features["team_elo"] = self.elo_ratings.get(int(team_id), 1500) if pd.notna(team_id) else 1500
        features["opponent_elo"] = self.elo_ratings.get(int(opponent_id), 1500) if pd.notna(opponent_id) else 1500
        features["elo_difference"] = features["team_elo"] - features["opponent_elo"]
        
        return features
    
    def update_elo_ratings(self, match_idx: int):
        """Update Elo ratings after a match result"""
        match = self.matches_df.iloc[match_idx]
        
        if pd.isna(match["result"]):
            return  # Match not played yet
        
        home_team_id = int(match["home_team_id"])
        away_team_id = int(match["away_team_id"])
        result = match["result"]
        
        home_elo = self.elo_ratings.get(home_team_id, 1500)
        away_elo = self.elo_ratings.get(away_team_id, 1500)
        
        expected_home = self._expected_score_elo(home_elo, away_elo)
        expected_away = 1 - expected_home
        
        if result == "H":
            actual_home, actual_away = 1, 0
        elif result == "A":
            actual_home, actual_away = 0, 1
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5
        
        home_change = self._calculate_elo_change(expected_home, actual_home)
        away_change = self._calculate_elo_change(expected_away, actual_away)
        
        self.elo_ratings[home_team_id] = home_elo + home_change
        self.elo_ratings[away_team_id] = away_elo + away_change

def prepare_training_data(matches_df: pd.DataFrame, split_date: str = None) -> pd.DataFrame:
    """
    Prepare training dataset with computed features.
    
    Args:
        matches_df: Raw matches dataframe
        split_date: ISO date string to split train/test
    
    Returns:
        features_df with all features and labels
    """
    logger.info("Preparing training data with feature engineering...")
    
    engineer = FeatureEngineer(matches_df)
    
    all_features = []
    all_labels = []
    
    for idx in range(len(matches_df)):
        try:
            home_features, away_features = engineer.compute_all_features(idx)
            engineer.update_elo_ratings(idx)
            
            match = matches_df.iloc[idx]
            
            if not pd.isna(match["result"]):  # Only for completed matches
                all_features.append({
                    "match_id": match["match_id"],
                    "date": match["date"],
                    "home_team_id": match["home_team_id"],
                    "away_team_id": match["away_team_id"],
                    **{f"home_{k}": v for k, v in home_features.items()},
                    **{f"away_{k}": v for k, v in away_features.items()},
                })
                all_labels.append(match["result"])
        
        except Exception as e:
            logger.warning(f"Error computing features for match {idx}: {e}")
            continue
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1} / {len(matches_df)} matches")
    
    features_df = pd.DataFrame(all_features)
    features_df["label"] = all_labels
    
    logger.info(f"Created {len(features_df)} feature vectors with {len(features_df.columns) - 5} features")
    
    return features_df