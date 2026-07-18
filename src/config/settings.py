from pathlib import Path
import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")
    config_path: Path = Field(default=Path("config/default.yaml"))

    @field_validator("openai_api_key")
    @classmethod
    def require_key(cls, v):
        if not v:
            raise ValueError("OPENAI_API_KEY required")
        return v

    def load_yaml(self):
        with open(self.config_path) as f:
            return yaml.safe_load(f)


_settings = None


def get_settings():
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
