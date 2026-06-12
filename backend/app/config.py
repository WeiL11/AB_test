from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/ab_testing"
    REDIS_URL: str = "redis://localhost:6379/0"  # default: no auth for local dev
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    DEFAULT_ALPHA: float = 0.05
    DEFAULT_POWER: float = 0.80
    MAX_BATCH_SIZE: int = 1000

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
    }


settings = Settings()
