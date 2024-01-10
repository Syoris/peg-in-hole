from pydantic_settings import BaseSettings
from pydantic import DirectoryPath, Field
from pathlib import Path


class Settings(BaseSettings):
    vortex_installation_path: DirectoryPath = Field(to_lower=True)


APP_SETTINGS = Settings()
