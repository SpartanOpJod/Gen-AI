from loguru import logger
from pathlib import Path
from .config import settings

LOG_PATH = Path("logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(LOG_PATH / "app.log", rotation="10 MB", retention="10 days", level=settings.log_level, enqueue=True)
logger.add(lambda msg: print(msg, end=""), level=settings.log_level)
