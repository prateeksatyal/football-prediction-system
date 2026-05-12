"""
Utility functions for logging, rate limiting, and helpers.
"""
import logging
import time
import asyncio
from functools import wraps
from datetime import datetime, timedelta
import pytz

# Setup logging
def setup_logging(level=logging.INFO):
    """Configure logging"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/football_prediction.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter with async support"""
    
    def __init__(self, min_delay: float = 1.0):
        self.min_delay = min_delay
        self.last_request_time = 0
    
    async def wait(self):
        """Wait until enough time has passed since last request"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            await asyncio.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()

def retry_async(max_retries: int = 3, delay: float = 2.0):
    """Decorator for async functions with retry logic"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator

def get_utc_now() -> datetime:
    """Get current UTC time"""
    return datetime.now(pytz.UTC)

def parse_match_date(date_str: str) -> datetime:
    """Parse match date string (ISO format)"""
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))

def days_between(date1: datetime, date2: datetime) -> int:
    """Calculate days between two dates"""
    return abs((date2 - date1).days)