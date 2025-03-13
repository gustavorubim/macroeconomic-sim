"""
Configuration package for the DSGE model.
"""

from config.default_config import (
    get_default_config,
    update_config,
    DEFAULT_CONFIG,
)
from config.config_manager import ConfigManager

__all__ = [
    "get_default_config",
    "update_config",
    "DEFAULT_CONFIG",
    "ConfigManager",
]