import sys

from pydantic_settings import BaseSettings
from pydantic import DirectoryPath, Field, field_validator, ValidationInfo
from pathlib import Path
import logging.config


def get_package_dir() -> Path:
    import peg_in_hole as ph

    package_path = Path(ph.__file__)

    package_dir = package_path.parent.parent

    return package_dir


class Settings(BaseSettings):
    package_path: DirectoryPath = get_package_dir()
    vortex_installation_path: DirectoryPath = Field(alias='vortex_path', to_lower=True)
    cfg_path: DirectoryPath = ''
    assets_path: DirectoryPath = ''
    vortex_resources_path: DirectoryPath = ''
    data_path: DirectoryPath = ''

    @field_validator('vortex_installation_path')
    @classmethod
    def add_vortex_to_python_path(cls, v: DirectoryPath) -> DirectoryPath:
        sys.path.append(str(v))
        sys.path.append(str(v / 'bin'))

        return v

    @field_validator('cfg_path', 'vortex_resources_path', 'assets_path', 'data_path')
    @classmethod
    def get_cfg_path(cls, v, info: ValidationInfo) -> DirectoryPath:
        pkg_path = info.data['package_path']

        if info.field_name == 'cfg_path':
            v = pkg_path / 'cfg'

        elif info.field_name == 'assets_path':
            v = pkg_path / 'assets'

        elif info.field_name == 'vortex_resources_path':
            assets_path = info.data['assets_path']
            v = assets_path / 'vortex'

        elif info.field_name == 'data_path':
            v = pkg_path / 'data'

        return v


app_settings = Settings()
logging.config.fileConfig(app_settings.cfg_path / 'logging.ini', disable_existing_loggers=False)
