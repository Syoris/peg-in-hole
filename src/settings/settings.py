import sys

from pydantic_settings import BaseSettings
from pydantic import DirectoryPath, Field, field_validator


class Settings(BaseSettings):
    vortex_installation_path: DirectoryPath = Field(to_lower=True)

    @field_validator('vortex_installation_path')
    @classmethod
    def add_vortex_to_python_path(cls, v: DirectoryPath) -> DirectoryPath:
        sys.path.append(str(v))
        sys.path.append(str(v / 'bin'))

        return v


APP_SETTINGS = Settings()
