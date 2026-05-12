"""
Data ingestion from legal free APIs:
1. Football-Data.org (free tier)
2. API-Football (free tier)
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import dict, list
import time

from .config import (
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_API_KEY,
    TARGET_LEAGUES,
    TARGET_SEASONS,
    RATE_LIMIT_DELAY,
    MAX_RETRIES,
    RETRY_DELAY,
)
from .database import db
from .utils import RateLimiter, retry_async, parse_match_date

logger = logging.getLogger(__name__)

class FootballDataOrgClient:
    """
    Client for football-data.org API (free tier allows 10 requests/min)
    Register at: https://www.football-data.org/client/register
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or FOOTBALL_DATA_API_KEY
        self.base_url = FOOTBALL_DATA_BASE_URL
        self.rate_limiter = RateLimiter(min_delay=RATE_LIMIT_DELAY)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    @retry_async(max_retries=MAX_RETRIES, delay=RETRY_DELAY)
    async def _get(self, endpoint: str, params: dict = None):
        """Make GET request with rate limiting and retries"""
        await self.rate_limiter.wait()
        
        headers = {"X-Auth-Token": self.api_key}
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"API error {response.status}: {await response.text()}")
    
    async def get_competitions(self) -> list:
        """Get available competitions"""
        logger.info("Fetching competitions...")
        data = await self._get("/competitions")
        return data.get("competitions", [])
    
    async def get_matches_for_league(self, league_code: str, season: str = None) -> list:
        """Get matches for a specific league and season"""
        logger.info(f"Fetching matches for {league_code} season {season}...")
        
        params = {}
        if season:
            params["season"] = season
        
        data = await self._get(f"/competitions/{league_code}/matches", params=params)
        return data.get("matches", [])
    
    async def get_standings(self, league_code: str, season: str = None) -> list:
        """Get league standings"""
        logger.info(f"Fetching standings for {league_code}...")
        
        params = {}
        if season:
            params["season"] = season
        
        data = await self._get(f"/competitions/{league_code}/standings", params=params)
        return data.get("standings", [])

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self):
        self.db = db
        self.teams_map = {}  # Local cache of team_id mappings
    
    async def fetch_and_ingest_all_data(self):
        """Main entry point: fetch all data and store in DB"""
        logger.info("Starting data pipeline...")
        
        async with FootballDataOrgClient() as client:
            for league_code, league_name in TARGET_LEAGUES.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {league_name} ({league_code})")
                logger.info(f"{'='*60}")
                
                for season in TARGET_SEASONS:
                    try:
                        matches = await client.get_matches_for_league(league_code, season)
                        await self._process_matches(matches, league_code, league_name, season)
                    except Exception as e:
                        logger.error(f"Error fetching {league_code} season {season}: {e}")
                        continue
        
        logger.info("Data pipeline completed!")
    
    async def _process_matches(self, matches: list, league_code: str, 
                               league_name: str, season: str):
        """Process and store matches from API"""
        teams_data = []
        matches_data = []
        
        for match in matches:
            try:
                # Extract teams
                home_team = match["homeTeam"]
                away_team = match["awayTeam"]
                
                home_team_name = home_team["name"]
                away_team_name = away_team["name"]
                
                # Store team info
                teams_data.append({
                    "team_name": home_team_name,
                    "league": league_name,
                    "country": home_team.get("area", {}).get("name", ""),
                })
                teams_data.append({
                    "team_name": away_team_name,
                    "league": league_name,
                    "country": away_team.get("area", {}).get("name", ""),
                })
                
                # Determine result
                status = match["status"]
                result = None
                home_goals = None
                away_goals = None
                
                if status in ["FINISHED", "LIVE"]:
                    home_goals = match["score"]["fullTime"]["home"]
                    away_goals = match["score"]["fullTime"]["away"]
                    
                    if home_goals > away_goals:
                        result = "H"
                    elif home_goals < away_goals:
                        result = "A"
                    else:
                        result = "D"
                
                # Store match info
                matches_data.append({
                    "api_match_id": f"fdo_{match['id']}",
                    "date": parse_match_date(match["utcDate"]),
                    "home_team_id": None,  # Will be resolved after team insertion
                    "away_team_id": None,
                    "league": league_name,
                    "season": season,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                    "odds_home": None,
                    "odds_draw": None,
                    "odds_away": None,
                })
            except Exception as e:
                logger.warning(f"Error processing match {match.get('id')}: {e}")
                continue
        
        # Insert teams
        if teams_data:
            self.db.insert_teams(teams_data)
        
        # Resolve team IDs and insert matches
        if matches_data:
            for idx, match_info in enumerate(matches_data):
                if idx < len(matches):
                    home_team_name = matches[idx]["homeTeam"]["name"]
                    away_team_name = matches[idx]["awayTeam"]["name"]
                    
                    home_id = self.db.get_team_id(home_team_name)
                    away_id = self.db.get_team_id(away_team_name)
                    
                    match_info["home_team_id"] = home_id
                    match_info["away_team_id"] = away_id
            
            self.db.insert_matches(matches_data)
        
        logger.info(f"Processed {len(matches)} matches for {league_name} {season}")

async def run_pipeline():
    """Run the full data pipeline"""
    pipeline = DataPipeline()
    await pipeline.fetch_and_ingest_all_data()

if __name__ == "__main__":
    asyncio.run(run_pipeline())