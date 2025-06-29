"""Configuration parser for AirflowLLM enterprise environments."""

import os
from typing import Any

import yaml


def parse_config_yaml(config_file_path: str) -> Any:
    """Parse environment-aware configuration YAML.

    Args:
        config_file_path: Path to the configuration YAML file

    Returns:
        Parsed configuration with environment-specific values resolved
    """
    deployment_environment = os.getenv("DEPLOYMENT_ENVIRONMENT", default="dev")

    with open(config_file_path) as file_data:
        env_config = yaml.safe_load(file_data)

        for key, value in env_config.items():
            if isinstance(value, dict) and deployment_environment in value:
                env_config[key] = value[deployment_environment]

    return env_config
