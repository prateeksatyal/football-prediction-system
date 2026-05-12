"""
Database management using SQLAlchemy with SQLite.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.engine import Engine
import logging

from .config import DB_PATH

logger = logging.getLogger(__name__)

Base = declarative_base()

class Team(Base):
    """Teams table"""
    __tablename__ = "teams"
    
    team_id = Column(Integer, primary_key=True)
    team_name = Column(String(100), unique=True, nullable=False)
    league = Column(String(50))
    country = Column(String(50))

class Match(Base):
    """Matches table"""
    __tablename__ = "matches"
    
    match_id = Column(Integer, primary_key=True)
    api_match_id = Column(String(50), unique=True)
    date = Column(DateTime, nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.team_id"))
    away_team_id = Column(Integer, ForeignKey("teams.team_id"))
    league = Column(String(50), nullable=False)
    season = Column(String(10))
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    result = Column(String(1))  # H, D, A
    odds_home = Column(Float)
    odds_draw = Column(Float)
    odds_away = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FeaturesCache(Base):
    """Pre-computed features cache"""
    __tablename__ = "features_cache"
    
    feature_id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"))
    team_id = Column(Integer, ForeignKey("teams.team_id"))
    feature_set = Column(JSON)  # Stores all features as JSON
    computed_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    """Model predictions"""
    __tablename__ = "predictions"
    
    prediction_id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"))
    model_version = Column(String(50))
    pred_home_prob = Column(Float)
    pred_draw_prob = Column(Float)
    pred_away_prob = Column(Float)
    predicted_result = Column(String(1))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Database:
    """Database connection and operations manager"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.engine = self._create_engine()
        self.Session = sessionmaker(bind=self.engine)
        self._initialize_db()
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine"""
        db_url = f"sqlite:///{self.db_path}"
        return create_engine(db_url, echo=False)
    
    def _initialize_db(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_session(self):
        """Get a new session"""
        return self.Session()
    
    def insert_teams(self, teams_data: list):
        """Insert teams (upsert behavior)"""
        session = self.get_session()
        try:
            for team_info in teams_data:
                existing = session.query(Team).filter_by(
                    team_name=team_info["team_name"]
                ).first()
                
                if not existing:
                    team = Team(
                        team_name=team_info["team_name"],
                        league=team_info.get("league"),
                        country=team_info.get("country"),
                    )
                    session.add(team)
            
            session.commit()
            logger.info(f"Inserted/updated {len(teams_data)} teams")
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting teams: {e}")
            raise
        finally:
            session.close()
    
    def insert_matches(self, matches_data: list):
        """Insert matches (upsert behavior)"""
        session = self.get_session()
        try:
            for match_info in matches_data:
                existing = session.query(Match).filter_by(
                    api_match_id=match_info["api_match_id"]
                ).first()
                
                if existing:
                    # Update if already exists
                    for key, value in match_info.items():
                        if key != "api_match_id":
                            setattr(existing, key, value)
                else:
                    match = Match(**match_info)
                    session.add(match)
            
            session.commit()
            logger.info(f"Inserted/updated {len(matches_data)} matches")
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting matches: {e}")
            raise
        finally:
            session.close()
    
    def get_matches_for_team(self, team_id: int, limit: int = None) -> pd.DataFrame:
        """Get all matches for a team"""
        session = self.get_session()
        try:
            query = session.query(Match).filter(
                (Match.home_team_id == team_id) | (Match.away_team_id == team_id)
            ).order_by(Match.date.desc())
            
            if limit:
                query = query.limit(limit)
            
            return pd.read_sql(query.statement, session.bind)
        finally:
            session.close()
    
    def get_all_completed_matches(self) -> pd.DataFrame:
        """Get all matches with results"""
        session = self.get_session()
        try:
            query = session.query(Match).filter(Match.result.isnot(None)).order_by(Match.date)
            return pd.read_sql(query.statement, session.bind)
        finally:
            session.close()
    
    def get_upcoming_matches(self) -> pd.DataFrame:
        """Get all matches without results"""
        session = self.get_session()
        try:
            query = session.query(Match).filter(Match.result.is_(None)).order_by(Match.date)
            return pd.read_sql(query.statement, session.bind)
        finally:
            session.close()
    
    def get_team_id(self, team_name: str) -> int:
        """Get team ID by name"""
        session = self.get_session()
        try:
            team = session.query(Team).filter_by(team_name=team_name).first()
            return team.team_id if team else None
        finally:
            session.close()
    
    def cache_features(self, match_id: int, team_id: int, features: dict):
        """Cache computed features"""
        session = self.get_session()
        try:
            cache_entry = FeaturesCache(
                match_id=match_id,
                team_id=team_id,
                feature_set=features,
            )
            session.add(cache_entry)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error caching features: {e}")
        finally:
            session.close()
    
    def get_cached_features(self, match_id: int, team_id: int) -> dict:
        """Get cached features"""
        session = self.get_session()
        try:
            cache = session.query(FeaturesCache).filter_by(
                match_id=match_id,
                team_id=team_id,
            ).first()
            return cache.feature_set if cache else None
        finally:
            session.close()

# Global database instance
db = Database()