import sys

from pydantic_settings import BaseSettings
from pydantic import DirectoryPath, Field, field_validator, ValidationInfo
from pathlib import Path


def get_package_dir() -> Path:
    import peg_in_hole as ph

    package_path = Path(ph.__file__)

    package_dir = package_path.parent.parent

    return package_dir


class Settings(BaseSettings):
    package_path: DirectoryPath = get_package_dir()
    vortex_installation_path: DirectoryPath = Field(to_lower=True)
    cfg_path: DirectoryPath = ''
    vortex_resources_path: DirectoryPath = ''

    @field_validator('vortex_installation_path')
    @classmethod
    def add_vortex_to_python_path(cls, v: DirectoryPath) -> DirectoryPath:
        sys.path.append(str(v))
        sys.path.append(str(v / 'bin'))

        return v

    @field_validator('cfg_path', 'vortex_resources_path')
    @classmethod
    def get_cfg_path(cls, v, info: ValidationInfo) -> DirectoryPath:
        pkg_path = info.data['package_path']

        if info.field_name == 'cfg_path':
            v = pkg_path / 'cfg'

        else:
            v = pkg_path / 'vortex_resources'

        return v


app_settings = Settings()
