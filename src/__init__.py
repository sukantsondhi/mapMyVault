"""
mapMyVault - Map files and create Obsidian vaults with AI metadata
"""

__version__ = "2.0.0"

from .config import MapperConfig
from .mapper import RepositoryMapper
from .query import VaultIndex

__all__ = ["MapperConfig", "RepositoryMapper", "VaultIndex"]
