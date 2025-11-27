import json
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    azure_openai_api_key: str | None = Field(None, env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str | None = Field(None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str | None = Field(
        None, env="AZURE_OPENAI_API_VERSION"
    )
    azure_openai_deployment: str | None = Field(None, env="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embeddings_deployment: str | None = Field(
        None, env="AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"
    )
    filesystem_root: str | None = Field(None, env="FILESYSTEM_ROOT")

    class Config:
        env_file = ".env"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()


def get_servers_config():
    config_path = Path(__file__).resolve().parent / "../Servers/servers_config.json"
    with open(config_path) as f:
        return json.load(f)
