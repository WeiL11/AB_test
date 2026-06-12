from functools import lru_cache

from app.config import Settings
from app.db import get_db  # noqa: F401 – re-exported for convenience


@lru_cache
def get_settings() -> Settings:
    return Settings()
