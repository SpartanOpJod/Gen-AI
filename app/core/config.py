from pydantic import BaseSettings, Field
from typing import Optional


class Settings(BaseSettings):
    app_env: str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    lancedb_dir: str = Field("./data/lancedb", env="LANCEDB_DIR")
    embedding_model: str = Field("BAAI/bge-small-en-v1.5", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(1024, env="EMBEDDING_DIM")
    similarity_threshold: float = Field(0.75, env="SIMILARITY_THRESHOLD")
    default_chunk_size: int = Field(500, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(100, env="DEFAULT_CHUNK_OVERLAP")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")

    class Config:
        env_file = ".env"


settings = Settings()
