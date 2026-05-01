"""Centralized configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # Garmin Connect
    garmin_email: str = Field(default="", description="Garmin Connect email")
    garmin_password: str = Field(default="", description="Garmin Connect password")
    garmin_token_dir: str = Field(
        default="~/.garminconnect",
        description="Directory to store Garmin auth tokens",
    )

    # LLM
    llm_provider: str = Field(
        default="gemini", description="LLM provider: 'gemini', 'groq', 'openai', or 'bedrock'"
    )
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name")
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(default="llama-3.1-70b-versatile", description="Groq model name")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    aws_region: str = Field(default="eu-west-1", description="AWS region for Bedrock")
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Bedrock model ID",
    )

    # Storage
    storage_backend: str = Field(
        default="local", description="Storage backend: 'local' or 'cloud'"
    )
    local_db_path: str = Field(
        default="data/coach.db", description="Path to local SQLite database"
    )

    # General
    log_level: str = Field(default="INFO", description="Logging level")
    lookback_days: int = Field(
        default=30, description="Default number of days to look back for data"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
